#include "tokenizer.hpp"
#include "utf8.h"
#include <algorithm>
#include <array>
#include <fstream>
#include <cstring>
#include <iostream>
#include <unordered_map>

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

namespace qwen {

namespace {

// 轮末/图像占位标签名是模型专属的字面量。
const char constexpr kChatMlTurnEnd[] = "<|im_end|>";
const char constexpr kPhiTurnEnd[] = "<|end|>";
// GLM-Edge/Llama 那一族（ChatML 早期壳）没有轮末标签：一轮由「下一个角色标签」
// 收尾，所以 generation_config 的 eos_token_id 是三个一串。字面量在这里按名查，
// 不写死 id（换 checkpoint 即失效）。
const char constexpr kGlmEndOfText[] = "<|endoftext|>";
const char constexpr kGlmUserTag[] = "<|user|>";
const char constexpr kGlmObservationTag[] = "<|observation|>";
const char constexpr kImagePadTag[] = "<|image_pad|>";

// tokenizer.bin header 的 flags 位（写端 tokenizer_to_bin.py 从 tokenizer.json 的
// pre_tokenizer 正则与 normalizer 自动推导）。0 = 既有 Qwen 口径。
constexpr uint32_t kFlagPhiPreTokenizer = 1u << 0;
constexpr uint32_t kFlagNoNormalizer = 1u << 1;
// GLM-Edge / Llama-3 那一族是 GPT-2 扫描器 + \p{N}{1,3}：只有数字段切成 1~3 个，
// 字母仍按 \p{L}+ 整段切（Phi 的 UL/LL 大小写形状切分不适用于它们）。
constexpr uint32_t kFlagDigitRun3 = 1u << 2;

// GPT-2/BBPE byte<->unicode 映射。68 个不可打印/特殊字节被映射到 U+0100..U+017F
// 区间（UTF-8 编码为 0xC4 0x80..0xC4 0xBF），其余字节一对一映射到自身码点。
// 因此 vocab 里的 token 字符串就是这些 codepoint 的 UTF-8 编码；encode 时把
// 输入的每个字节转成对应的 UTF-8 字符串，decode 时把 token 字符串的每个
// codepoint 还原回字节。
struct ByteUnicodeMap {
    std::array<std::string, 256> byte_to_str;        // 字节 -> UTF-8 字符串
    std::unordered_map<uint32_t, uint8_t> cp_to_byte; // codepoint -> 字节
    ByteUnicodeMap() {
        std::vector<uint32_t> bs;
        for (uint32_t i = '!'; i <= '~'; ++i) bs.push_back(i);
        for (uint32_t i = 0xA1; i <= 0xAC; ++i) bs.push_back(i);
        for (uint32_t i = 0xAE; i <= 0xFF; ++i) bs.push_back(i);
        std::vector<bool> in_bs(256, false);
        for (auto c : bs) in_bs[c] = true;

        uint32_t n = 0;
        for (uint32_t b = 0; b < 256; ++b) {
            uint32_t cp;
            if (in_bs[b]) {
                cp = b;
            } else {
                cp = 256 + n;
                ++n;
            }
            // codepoint -> UTF-8 string
            std::string s;
            if (cp < 0x80) {
                s += static_cast<char>(cp);
            } else if (cp < 0x800) {
                s += static_cast<char>(0xC0 | (cp >> 6));
                s += static_cast<char>(0x80 | (cp & 0x3F));
            } else {
                s += static_cast<char>(0xE0 | (cp >> 12));
                s += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
                s += static_cast<char>(0x80 | (cp & 0x3F));
            }
            byte_to_str[b] = s;
            cp_to_byte[cp] = static_cast<uint8_t>(b);
        }
    }
};

const ByteUnicodeMap& byte_unicode_map() {
    static const ByteUnicodeMap m;
    return m;
}

// ---- Unicode 类别判定（用于 GPT-2 pre_tokenizer 正则的 \p{L}/\p{N}/\s）----
// 用 utf8proc 的类别枚举：L=Lu/Ll/Lt/Lm/Lo(1-5)，N=Nd/Nl/No(9-11)，
// \s = ASCII 空白(含 \x85) + Zs/Zl/Zp(23-25)。与 Rust regex 的 \p{L}/\p{N}/\s
// 语义一致（已对 HF tokenizers 实测验证）。
inline bool is_letter(uint32_t cp) {
    auto c = utf8proc_category(cp);
    return c == UTF8PROC_CATEGORY_LU || c == UTF8PROC_CATEGORY_LL
        || c == UTF8PROC_CATEGORY_LT || c == UTF8PROC_CATEGORY_LM
        || c == UTF8PROC_CATEGORY_LO;
}

inline bool is_number(uint32_t cp) {
    auto c = utf8proc_category(cp);
    return c == UTF8PROC_CATEGORY_ND || c == UTF8PROC_CATEGORY_NL
        || c == UTF8PROC_CATEGORY_NO;
}

// Phi-4 的 pre_tokenizer 把字母按「形状」分两类（\p{M} 组合记号两边都算）：
//   UL = \p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}   大写/形首
//   LL = \p{Ll}\p{Lm}\p{Lo}\p{M}         小写/形尾
// Lm/Lo/M 同属两类，这正是 [UL]*[LL]+ 需要回退的地方（见 pre_tokenize_phi）。
inline bool is_upper_class(uint32_t cp) {
    auto c = utf8proc_category(cp);
    return c == UTF8PROC_CATEGORY_LU || c == UTF8PROC_CATEGORY_LT
        || c == UTF8PROC_CATEGORY_LM || c == UTF8PROC_CATEGORY_LO
        || c == UTF8PROC_CATEGORY_ME || c == UTF8PROC_CATEGORY_MN
        || c == UTF8PROC_CATEGORY_MC;
}

inline bool is_lower_class(uint32_t cp) {
    auto c = utf8proc_category(cp);
    return c == UTF8PROC_CATEGORY_LL || c == UTF8PROC_CATEGORY_LM
        || c == UTF8PROC_CATEGORY_LO || c == UTF8PROC_CATEGORY_ME
        || c == UTF8PROC_CATEGORY_MN || c == UTF8PROC_CATEGORY_MC;
}

inline bool is_ws(uint32_t cp) {
    switch (cp) {
    case 0x09: case 0x0A: case 0x0B: case 0x0C: case 0x0D: case 0x20: case 0x85:
        return true;
    default: {
        auto c = utf8proc_category(cp);
        return c == UTF8PROC_CATEGORY_ZS || c == UTF8PROC_CATEGORY_ZL
            || c == UTF8PROC_CATEGORY_ZP;
    }
    }
}

inline int get_utf8_char_len(unsigned char c) {
    if (c < 0x80) return 1;
    else if ((c >> 5) == 0x06) return 2;
    else if ((c >> 4) == 0x0E) return 3;
    else if ((c >> 3) == 0x1E) return 4;
    return 1; // 遇到非法 UTF-8 字节，按单字节处理
}

// 解码一个 UTF-8 码点，返回码点值并前进指针。
inline uint32_t decode_utf8(const char*& p, const char* end) {
    unsigned char c = static_cast<unsigned char>(*p);
    if (c < 0x80) { uint32_t cp = c; p += 1; return cp; }
    int len = get_utf8_char_len(c);
    if (p + len > end) { uint32_t cp = c; p += 1; return cp; }
    uint32_t cp = c & ((1 << (7 - len)) - 1);
    for (int i = 1; i < len; ++i) {
        cp = (cp << 6) | (static_cast<unsigned char>(p[i]) & 0x3F);
    }
    p += len;
    return cp;
}

// 把字符串解码成 codepoint 数组，并记录每个 codepoint 在原字节串的起始偏移
// （pre_tokenize 切片时用偏移切回 UTF-8 子串）。
struct CpStream {
    std::vector<uint32_t> cps;          // codepoint 序列
    std::vector<std::size_t> offsets;   // cps[i] 在原 s 的字节起始偏移；末尾多一个 = s.size()
};

CpStream decode_to_cps(const std::string& s) {
    CpStream st;
    st.cps.reserve(s.size());
    st.offsets.reserve(s.size() + 1);
    const char* p = s.data();
    const char* end = p + s.size();
    while (p < end) {
        st.offsets.push_back(static_cast<std::size_t>(p - s.data()));
        st.cps.push_back(decode_utf8(p, end));
    }
    st.offsets.push_back(s.size());
    return st;
}

// 缩写后缀 (?i:'s|'t|'re|'ve|'m|'ll|'d)：返回从 i 起匹配的 codepoint 数，
// i 处不是撇号（或后面接不上表里的词）就返回 0。GPT-2 把它当独立分支，Phi-4 只挂在
// 字母段尾巴上 —— 尾巴那边调用点不知道下一个字符是什么，所以撇号必须在这里查：
// 少了这道判断，" leading spaces" 会被当成「空格 + s」的缩写，把后面的 ' s' 吞进
// 前一个 span。
std::size_t match_contraction(const std::vector<uint32_t>& cps, std::size_t i) {
    std::size_t n = cps.size();
    if (i >= n || cps[i] != 0x27) return 0;  // 0x27 = '\''
    if (i + 1 >= n) return 0;
    uint32_t c = cps[i + 1];
    auto lower = [](uint32_t ch) { return (ch >= 'A' && ch <= 'Z') ? ch + 32 : ch; };
    uint32_t l = lower(c);
    if (l == 's' || l == 't' || l == 'm' || l == 'd') return 2;
    if (l == 'r' || l == 'v') {
        if (i + 2 < n && lower(cps[i + 2]) == 'e') return 3;
        return 0;
    }
    if (l == 'l') {
        if (i + 2 < n && lower(cps[i + 2]) == 'l') return 3;
        return 0;
    }
    return 0;
}

// GPT-2 pre_tokenizer 正则的手写实现。HF 原版（tokenizer.json）正则为：
//   (?i:'s|'t|'re|'ve|'m|'ll|'d)              alt1
//   | [^\r\n\p{L}\p{N}]? \p{L}+               alt2
//   | \p{N}{1,digit_max}                       alt3   Qwen=1，GLM/Llama3=3
//   | ' ?[^\s\p{L}\p{N}]+ [\r\n]*              alt4
//   | \s* [\r\n]+                              alt5
//   | \s+ (?!\S)                               alt6  (RE2 不支持前瞻，手写复刻)
//   | \s+                                      alt7
// 关键点（已用 HF tokenizers + PyPI regex 库双重验证）：
//  - alt2 前缀 [^\r\n\p{L}\p{N}]? 可吞一个非(CR/NL/字母/数字)字符（空格/Tab/标点）。
//  - alt4 前缀 ' ? 仅字面空格 0x20。
//  - alt5: \s-run 内含 CR/NL 时，吞 [i..最后一个CR/NL+1)，trailing 空白另起。
//  - alt6: \s-run 末尾若紧跟非空白，回溯留最后一个 \s 给后续 alt2/alt4 吸附；
//    run==1 且无法吸附（如后接数字）时 alt6 失败，由 alt7 吞。
//  - alt3 的数字段长度由 digit_max 参数决定：Qwen3-VL 写 \p{N}（=1），GLM-Edge 与
//   Llama-3 写 \p{N}{1,3}。这条差异只影响 span 边界，不影响其余分支。

// \s-run 的三条分支在两个 pre_tokenizer 变体里完全一致，共享这一份：
//   \s*[\r\n]+   run 内含 CR/NL 时吞到最后一个 CR/NL（前导空白一起吞）
//   \s+(?!\S)    纯空白 run：到 EOF 吞满；后面还有非空白则回溯，留最后一个 \s
//                给下一轮带可选前缀的 alt2/alt4 吸附
//   \s+          run==1 且无处可吸附时整段吞掉
// 返回匹配区间 [i, end)，end 一定 > i。
std::size_t match_ws_run(const std::vector<uint32_t>& cps, std::size_t i) {
    const std::size_t n = cps.size();
    std::size_t j = i;
    while (j < n && is_ws(cps[j])) ++j;

    std::size_t last_crlf = std::size_t(-1);
    for (std::size_t k = i; k < j; ++k) {
        if (cps[k] == 0x0D || cps[k] == 0x0A) last_crlf = k;
    }
    if (last_crlf != std::size_t(-1)) return last_crlf + 1;

    if (j == n) return j;           // EOF：没有 \S 跟在后面，前瞻成立
    if (j - 1 > i) return j - 1;    // run>=2：留最后一个 \s 给后续吸附
    return j;                       // run==1：alt6 失败，退到 alt7
}

void pre_tokenize(const std::string& s, std::vector<std::string>& out,
                  std::size_t digit_max) {
    CpStream st = decode_to_cps(s);
    const auto& cps = st.cps;
    const auto& off = st.offsets;
    std::size_t n = cps.size();
    std::size_t i = 0;

    auto emit = [&](std::size_t a, std::size_t b) {
        // codepoint 区间 [a,b) -> 原字节子串
        out.emplace_back(s, off[a], off[b] - off[a]);
    };

    while (i < n) {
        uint32_t cp = cps[i];

        // alt1: 缩写
        if (cp == 0x27) { // apostrophe
            std::size_t m = match_contraction(cps, i);
            if (m > 0) { emit(i, i + m); i += m; continue; }
        }

        // alt2: [^\r\n\p{L}\p{N}]? \p{L}+
        //   无前缀：cp 是字母
        //   有前缀：cp 不是 CR/NL/L/N，且下一个是字母
        bool cp_letter = is_letter(cp);
        if (cp_letter
            || (cp != 0x0D && cp != 0x0A && !is_letter(cp) && !is_number(cp)
                && i + 1 < n && is_letter(cps[i + 1]))) {
            std::size_t j = cp_letter ? i : i + 1; // 前缀消费 0 或 1 个
            while (j < n && is_letter(cps[j])) ++j;
            emit(i, j); i = j; continue;
        }

        // alt3: \p{N}{1,digit_max}。Qwen3-VL 给 1（逐位切），GLM-Edge/Llama-3 给 3。
        if (is_number(cp)) {
            std::size_t j = i;
            while (j < n && is_number(cps[j]) && j - i < digit_max) ++j;
            emit(i, j); i = j; continue;
        }

        // alt4: ' ?[^\s\p{L}\p{N}]+ [\r\n]*  （前缀仅字面空格 0x20）
        bool cp_nonsln = !is_ws(cp) && !is_letter(cp) && !is_number(cp);
        if ((cp == 0x20 && i + 1 < n && !is_ws(cps[i + 1]) && !is_letter(cps[i + 1]) && !is_number(cps[i + 1]))
            || cp_nonsln) {
            std::size_t j = (cp == 0x20) ? i + 1 : i;
            while (j < n && !is_ws(cps[j]) && !is_letter(cps[j]) && !is_number(cps[j])) ++j; // [^\s\p{L}\p{N}]+
            while (j < n && (cps[j] == 0x0D || cps[j] == 0x0A)) ++j;                          // [\r\n]*
            emit(i, j); i = j; continue;
        }

        // 以下处理 \s（alt5/alt6/alt7，见 match_ws_run）。
        if (is_ws(cp)) {
            std::size_t j = match_ws_run(cps, i);
            emit(i, j); i = j; continue;
        }

        // 兜底：单 codepoint（理论上 GPT-2 正则覆盖所有输入，不会到这）
        emit(i, i + 1); i += 1;
    }
}

// ---- Phi-4 / o200k 变体的 pre_tokenizer ----
// tokenizer.json 里的 Split 正则（顺序即 Rust regex 的 leftmost-first 优先级）：
//   [^CR LF \pL \pN]? [UL]* [LL]+ (contr)?   alt1   小写结尾的单词（含全小写）
// | [^CR LF \pL \pN]? [UL]+ [LL]* (contr)?   alt2   大写开头的单词
// | \pN{1,3}                                  alt3   最多 3 个连续数字
// | ' ? [^ \t\n... \pL \pN]+ [CR LF /]*      alt4   标点串（可带一个前导空格）
// | \s* [CR LF]+                              alt5
// | \s+ (?!\S)                                alt6
// | \s+                                       alt7
// 其中 UL = Lu Lt Lm Lo M、LL = Ll Lm Lo M（Lm/Lo/M 同时属于两类），contr =
// (?i:'s|'t|'re|'ve|'m|'ll|'d)。与 GPT-2 的三处实质差异：
//  1. 字母按「大小写形状」切分：alt1 要求末尾段是小写类，alt2 要求开头段是大写
//     类，于是 "HelloWorld" 切成 "Hello" + "World"，"ABc" 是一个 token。
//     GPT-2 只有 \p{L}+，整串一个 token。
//  2. 数字是 \p{N}{1,3}（贪心 1~3 个），GPT-2 是单个 \p{N}。
//  3. 标点串的拖尾字符类多了一个 '/'，且缩写只能挂在单词尾巴上（GPT-2 有独立的
//     缩写分支，所以 "'s" 单独也能成 token）。
// 量化器的贪心与回退按 Rust regex 的语义复刻：[UL]* 先吃满，只有在 [LL]+ 无法满足
// 时才逐个回退 —— 这在 Lm/Lo/M（同时属于两类）连续出现时会改变匹配长度，例如
// "ᐁᐁ"（加拿大音节文字，Lo）要靠回退才能整体匹配。
void pre_tokenize_phi(const std::string& s, std::vector<std::string>& out) {
    CpStream st = decode_to_cps(s);
    const auto& cps = st.cps;
    const auto& off = st.offsets;
    const std::size_t n = cps.size();
    std::size_t i = 0;

    auto emit = [&](std::size_t a, std::size_t b) {
        out.emplace_back(s, off[a], off[b] - off[a]);
    };

    // 从 j 起匹配 [UL]*[LL]+（alt1）或 [UL]+[LL]*（alt2）；返回区间结束位置，
    // 不匹配返回 npos。调用方再决定能否吸附缩写后缀。
    auto letters = [&](std::size_t j, bool alt1) -> std::size_t {
        std::size_t u = j;
        while (u < n && is_upper_class(cps[u])) ++u;   // [UL]* / [UL]+ 贪心
        std::size_t start = u;
        if (alt1) {
            // [LL]+ 至少要 1 个：u 处是小写类就直接起步，否则把 [UL] 逐个回退，
            // 直到落在一个「同时属于两类」的字符上（Lm/Lo/M）；退到 j 仍无解则失败。
            for (;; --start) {
                if (start < n && is_lower_class(cps[start])) break;
                if (start == j) return std::size_t(-1);
            }
        } else if (u == j) {
            return std::size_t(-1);                    // [UL]+ 至少要 1 个
        }
        std::size_t t = start;
        while (t < n && is_lower_class(cps[t])) ++t;
        return t;
    };

    while (i < n) {
        const uint32_t cp = cps[i];

        // alt1 / alt2：可选前缀（非 CR/LF/字母/数字）+ 大小写形状 + 可选缩写。
        // 前缀是贪婪可选：先试「带前缀」，整条分支失败才试「不带前缀」。缩写后缀
        // 挂在字母段尾巴上；单独的 "'s" 也能匹配 —— 撇号正是那条可选前缀。
        bool prefix_ok = cp != 0x0D && cp != 0x0A && !is_letter(cp) && !is_number(cp);
        bool matched = false;
        for (bool alt1 : {true, false}) {
            std::size_t end = letters(i, alt1);
            if (prefix_ok) {
                std::size_t with_prefix = letters(i + 1, alt1);
                if (with_prefix != std::size_t(-1)) end = with_prefix;
            }
            if (end == std::size_t(-1)) continue;
            std::size_t m = match_contraction(cps, end);
            emit(i, end + m);
            i = end + m;
            matched = true;
            break;
        }
        if (matched) continue;

        // alt3: \p{N}{1,3}
        if (is_number(cp)) {
            std::size_t j = i;
            while (j < n && is_number(cps[j]) && j - i < 3) ++j;
            emit(i, j); i = j; continue;
        }

        // alt4: ' ?[^\s\p{L}\p{N}]+[\r\n/]*  （前缀仅字面空格 0x20）
        auto nonsln = [&](uint32_t c) {
            return !is_ws(c) && !is_letter(c) && !is_number(c);
        };
        if ((cp == 0x20 && i + 1 < n && nonsln(cps[i + 1])) || nonsln(cp)) {
            std::size_t j = (cp == 0x20) ? i + 1 : i;
            while (j < n && nonsln(cps[j])) ++j;
            while (j < n && (cps[j] == 0x0D || cps[j] == 0x0A || cps[j] == 0x2F)) ++j;
            emit(i, j); i = j; continue;
        }

        // alt5/alt6/alt7
        if (is_ws(cp)) {
            std::size_t j = match_ws_run(cps, i);
            emit(i, j); i = j; continue;
        }

        // 兜底：正则不覆盖的码点（例如落在两条分支缝隙里的字符）单独成 span。
        emit(i, i + 1); i += 1;
    }
}

} // namespace

Tokenizer::~Tokenizer() {
    if (mmap_data_ && mmap_data_ != MAP_FAILED) {
        munmap(mmap_data_, mmap_size_);
    }
}

namespace {

// 从 ptr 读一个 u16（小端）。
inline uint16_t read_u16(const uint8_t* p) {
    return static_cast<uint16_t>(p[0]) | (static_cast<uint16_t>(p[1]) << 8);
}
inline uint32_t read_u32(const uint8_t* p) {
    return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8)
         | (static_cast<uint32_t>(p[2]) << 16) | (static_cast<uint32_t>(p[3]) << 24);
}

// 从 mmap 区读取一个长度前缀字符串：u16 len + bytes。推进 ptr，越界抛异常。
std::string read_len_str(const uint8_t*& ptr, const uint8_t* end, const char* ctx) {
    if (ptr + 2 > end) throw std::runtime_error(std::string("truncated ") + ctx);
    uint16_t len = read_u16(ptr); ptr += 2;
    if (ptr + len > end) throw std::runtime_error(std::string("truncated ") + ctx + " entry");
    std::string s(reinterpret_cast<const char*>(ptr), len);
    ptr += len;
    return s;
}

} // namespace

void Tokenizer::load(const std::string& bin_path) {
    int fd = open(bin_path.c_str(), O_RDONLY);
    if (fd == -1) throw std::runtime_error("Failed to open bin file: " + bin_path);

    struct stat st;
    fstat(fd, &st);
    mmap_size_ = st.st_size;

    mmap_data_ = mmap(nullptr, mmap_size_, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (mmap_data_ == MAP_FAILED) throw std::runtime_error("mmap failed");

    const uint8_t* ptr = static_cast<const uint8_t*>(mmap_data_);
    const uint8_t* end = ptr + mmap_size_;
    if (mmap_size_ < 24 || std::memcmp(ptr, "QW3T", 4) != 0)
        throw std::runtime_error("Invalid magic number");
    ptr += 4;

    ptr += 4; // version
    vocab_size_ = read_u32(ptr); ptr += 4;
    uint32_t merge_count = read_u32(ptr); ptr += 4;
    uint32_t special_count = read_u32(ptr); ptr += 4;
    // 第 5 个 u32 原本是 Reserved（写端恒填 0），现在当 flags：
    //   bit0 = pre_tokenizer 走 Phi-4/o200k 变体，bit1 = tokenizer.json 没有
    //   normalizer（跳过 NFC），bit2 = GPT-2 扫描器的数字段切成 1~3 个。
    //   旧 bin 这里是 0 → 既有 Qwen 口径完全不变。
    flags_ = read_u32(ptr); ptr += 4;
    // 未定义的位一律拒绝：分词口径错了不会崩，只会静默切错 token，GPU 上跑完也
    // 看不出来，所以写端加了新 bit 而读端不认识时必须当场报错，不能当 0 处理。
    constexpr uint32_t kKnownFlags = kFlagPhiPreTokenizer | kFlagNoNormalizer
                                   | kFlagDigitRun3;
    if (flags_ & ~kKnownFlags)
        throw std::runtime_error("tokenizer.bin has unknown flag bits: "
                                 + std::to_string(flags_));

    std::cout << "[Tokenizer] Vocab: " << vocab_size_ << ", Merges: " << merge_count
              << ", Special: " << special_count
              << ", Flags: " << flags_
              << ((flags_ & kFlagPhiPreTokenizer) ? " (phi pre_tokenizer)" : "")
              << ((flags_ & kFlagDigitRun3) ? " (digits 1-3)" : "")
              << ((flags_ & kFlagNoNormalizer) ? " (no NFC)" : "") << "\n";

    // 解析 Vocab Section
    id_to_token_.resize(vocab_size_);
    token_to_id_.reserve(vocab_size_);
    for (uint32_t i = 0; i < vocab_size_; ++i) {
        if (ptr + 2 > end) throw std::runtime_error("truncated vocab");
        uint16_t len = read_u16(ptr); ptr += 2;
        if (ptr + len + 4 > end) throw std::runtime_error("truncated vocab entry");
        std::string token(reinterpret_cast<const char*>(ptr), len); ptr += len;
        uint32_t tid = read_u32(ptr); ptr += 4;
        if (tid >= vocab_size_) throw std::runtime_error("token id out of range");
        id_to_token_[tid] = std::move(token);
        token_to_id_[id_to_token_[tid]] = tid;
    }

    // 解析 Merge Rules Section. merge 顺序即 BPE 优先级（rank），bin 文件按
    // tokenizer.json 的 merges 顺序写入，所以 (left_id,right_id) 在哈希表里
    // 直接映射到 merged_id，rank 隐含在遍历顺序中——但 bpe_merge 需要按
    // rank 选最优 pair，所以这里同时建一张 (left,right)->rank 表。
    merge_map_.reserve(merge_count);
    merge_rank_.reserve(merge_count);
    for (uint32_t i = 0; i < merge_count; ++i) {
        if (ptr + 12 > end) throw std::runtime_error("truncated merge rules");
        uint32_t left = read_u32(ptr); ptr += 4;
        uint32_t right = read_u32(ptr); ptr += 4;
        uint32_t merged = read_u32(ptr); ptr += 4;
        if (left == 0 && right == 0 && merged == 0) {
            // 跳过无效规则占位（写端 skip 的行）。
            continue;
        }
        merge_map_[{left, right}] = merged;
        merge_rank_[{left, right}] = i;
    }

    // 解析 Special Tokens Section。这些 token（如 <|im_start|>）不在
    // model.vocab 里，id 紧接 vocab 区间。把它们并入 id_to_token_ /
    // token_to_id_，并单独存一份给 encode 做最长字面量匹配。
    for (uint32_t i = 0; i < special_count; ++i) {
        std::string content = read_len_str(ptr, end, "special tokens");
        if (ptr + 4 > end) throw std::runtime_error("truncated special entry");
        uint32_t sid = read_u32(ptr); ptr += 4;
        if (sid >= id_to_token_.size()) id_to_token_.resize(sid + 1);
        id_to_token_[sid] = content;
        token_to_id_[content] = sid;
        special_tokens_.push_back({std::move(content), sid});
    }
    total_size_ = static_cast<uint32_t>(id_to_token_.size());
    // 按字面量长度降序排列，保证 encode 时最长匹配优先。
    std::sort(special_tokens_.begin(), special_tokens_.end(),
              [](const SpecialToken& a, const SpecialToken& b) {
                  return a.content.size() > b.content.size();
              });

    std::cout << "[Tokenizer] Load successful.\n";

    // 解析可选的 ChatTemplate Section（文件剩余字节）。旧 bin 无此段时跳过，
    // chat_template_ 保持空，apply_chat_template 返回空串。
    if (ptr < end) {
        try {
            if (ptr + 4 <= end) {
                uint32_t role_count = read_u32(ptr); ptr += 4;
                for (uint32_t r = 0; r < role_count && ptr < end; ++r) {
                    std::string name = read_len_str(ptr, end, "chat role");
                    std::string prefix = read_len_str(ptr, end, "chat role prefix");
                    std::string suffix = read_len_str(ptr, end, "chat role suffix");
                    chat_template_.roles[std::move(name)] = {std::move(prefix), std::move(suffix)};
                }
                if (ptr + 4 <= end) {
                    uint32_t ct_count = read_u32(ptr); ptr += 4;
                    for (uint32_t c = 0; c < ct_count && ptr < end; ++c) {
                        std::string type = read_len_str(ptr, end, "chat content type");
                        std::string fmt = read_len_str(ptr, end, "chat content format");
                        chat_template_.content_types[std::move(type)] = std::move(fmt);
                    }
                }
                chat_template_.generation_prompt = read_len_str(ptr, end, "chat generation_prompt");
                chat_template_.default_system_prompt = read_len_str(ptr, end, "chat default_system_prompt");
                std::cout << "[Tokenizer] Chat template loaded: "
                          << chat_template_.roles.size() << " roles, "
                          << chat_template_.content_types.size() << " content types.\n";
            }
        } catch (const std::exception& e) {
            // 解析失败不阻断分词，仅清空 chat template。
            std::cerr << "[Tokenizer] Warning: failed to parse ChatTemplate section ("
                      << e.what() << "), chat template disabled.\n";
            chat_template_ = ChatTemplate{};
        }
    }
}

std::string Tokenizer::normalize_nfc(const std::string& text) const {
    // NFC = 先分解（NFD）再组合，输出「最短」等价形式。Qwen3-VL 的 vocab
    // 假设 NFC 输入，若直接喂 NFD（如 e + U+0301 而非 U+00E9），字节序列
    // 不同，BBPE 会切出不一样的 token。utf8proc_NFC 是 utf8proc_map 的快捷：
    // 输入 null-terminated UTF-8，输出 malloc 的新串，调用方负责 free。
    if (text.empty()) return text;
    utf8proc_uint8_t* dst = utf8proc_NFC(
        reinterpret_cast<const utf8proc_uint8_t*>(text.data()));
    if (!dst) {
        // 非法 UTF-8 或内存不足：退回原串，不阻断分词。
        return text;
    }
    std::string out(reinterpret_cast<const char*>(dst));
    free(dst);
    return out;
}

std::vector<uint32_t> Tokenizer::encode(const std::string& text) const {
    // pipeline: Normalizer(NFC) → special token 切分 → pre_tokenizer
    //           → ByteLevel + BPE。NFC 在最前，保证字节序列规范化后再切分。
    // 两处口径都来自 bin header 的 flags：Phi-4 的 tokenizer.json 没有 normalizer
    // （NFC 会把 é 这类「基码点+组合符」合成另一个码点，词表按原样建，必须跳过），
    // pre_tokenizer 正则也与 GPT-2 不同。
    std::string normalized = (flags_ & kFlagNoNormalizer) ? text : normalize_nfc(text);
    const std::string& input = normalized;

    std::vector<uint32_t> tokens;
    tokens.reserve(input.size());

    // 1. 特殊 token 字面量先切出（最长匹配，作为原子 id，不进 BPE）。
    // 2. 其余部分用 pre_tokenizer 切成 spans，每段独立做字节→BBPE + BPE，
    //    不跨段合并——这是和 HF 官方 tokenizer 对齐的关键。
    size_t i = 0;
    const size_t n = input.size();
    while (i < n) {
        // 尝试在当前位置匹配最长特殊 token。
        bool matched = false;
        for (const auto& sp : special_tokens_) {
            const auto& s = sp.content;
            if (s.size() <= n - i && std::memcmp(input.data() + i, s.data(), s.size()) == 0) {
                tokens.push_back(sp.id);
                i += s.size();
                matched = true;
                break;
            }
        }
        if (matched) continue;

        // 从 i 找到下一个特殊 token 出现位置 seg_end（不含），把 [i, seg_end)
        // 整段交给 pre_tokenize 一次切分，再逐 piece BPE。这样既避免 pre_tokenize
        // 把特殊 token 字面量拆开，又减少调用次数。
        size_t seg_end = n;
        for (const auto& sp : special_tokens_) {
            const auto& s = sp.content;
            if (s.empty() || s.size() > n - i) continue;
            // 在 [i+1, n-s.size()+1] 范围找 s 的最早出现
            size_t limit = n - s.size() + 1;
            const char* base = input.data() + i + 1;
            size_t search_n = (limit > i + 1) ? (limit - (i + 1)) : 0;
            if (search_n == 0) continue;
            const char* found = static_cast<const char*>(
                std::memchr(base, s[0], search_n));
            while (found) {
                size_t pos = static_cast<size_t>(found - input.data());
                if (std::memcmp(found, s.data(), s.size()) == 0) {
                    if (pos < seg_end) seg_end = pos;
                    break;
                }
                size_t consumed = static_cast<size_t>(found - base) + 1;
                size_t remain = search_n - consumed;
                if (remain == 0) break;
                found = static_cast<const char*>(std::memchr(base + consumed, s[0], remain));
                (void)pos;
            }
        }

        std::string seg(input, i, seg_end - i);
        std::vector<std::string> pieces;
        if (flags_ & kFlagPhiPreTokenizer) pre_tokenize_phi(seg, pieces);
        else pre_tokenize(seg, pieces, (flags_ & kFlagDigitRun3) ? 3 : 1);
        for (const auto& piece : pieces) {
            encode_segment(piece, tokens);
        }
        i = seg_end;
    }

    post_process(tokens);
    return tokens;
}

void Tokenizer::post_process(std::vector<uint32_t>& /*ids*/) const {
    // Qwen3-VL 的 ByteLevel post_processor: add_prefix_space=false,
    // trim_offsets=false, use_regex=false
    // means no-op
}

void Tokenizer::encode_segment(const std::string& seg, std::vector<uint32_t>& out) const {
    const auto& m = byte_unicode_map();
    // 把这段的每个字节映射成 BBPE token id，单独做 BPE 后再追加到 out。
    // 关键：每段独立 BPE，不跨 pre-tokenizer 边界合并——这是和 HF 对齐的核心。
    std::vector<uint32_t> seg_tokens;
    seg_tokens.reserve(seg.size());
    for (unsigned char c : seg) {
        auto it = token_to_id_.find(m.byte_to_str[c]);
        if (it != token_to_id_.end()) {
            seg_tokens.push_back(it->second);
        }
    }
    bpe_merge(seg_tokens);
    out.insert(out.end(), seg_tokens.begin(), seg_tokens.end());
}

void Tokenizer::bpe_merge(std::vector<uint32_t>& tokens) const {
    // 标准 BPE：每轮选取 rank 最小（优先级最高）的相邻 pair 合并，直到没有
    // 可合并的 pair。rank 来自 merge 顺序（tokenizer.json merges 的下标）。
    while (tokens.size() >= 2) {
        uint32_t best_rank = UINT32_MAX;
        size_t best_idx = SIZE_MAX;
        for (size_t i = 0; i + 1 < tokens.size(); ++i) {
            auto it = merge_rank_.find({tokens[i], tokens[i + 1]});
            if (it != merge_rank_.end() && it->second < best_rank) {
                best_rank = it->second;
                best_idx = i;
            }
        }
        if (best_idx == SIZE_MAX) break;
        auto mit = merge_map_.find({tokens[best_idx], tokens[best_idx + 1]});
        if (mit == merge_map_.end()) break;
        tokens[best_idx] = mit->second;
        tokens.erase(tokens.begin() + best_idx + 1);
    }
}

std::string Tokenizer::decode(const std::vector<uint32_t>& ids) const {
    const auto& m = byte_unicode_map();
    // 先把所有 token 字符串拼起来，再按 codepoint 还原回字节。codepoint 在
    // cp_to_byte 里的就是普通字节；不在里面的（特殊 token 等）按 UTF-8 透传。
    std::string joined;
    for (uint32_t id : ids) {
        if (id < total_size_) {
            joined += id_to_token_[id];
        }
    }
    std::string out;
    const char* p = joined.data();
    const char* end = p + joined.size();
    while (p < end) {
        const char* before = p;
        uint32_t cp = decode_utf8(p, end);
        auto it = m.cp_to_byte.find(cp);
        if (it != m.cp_to_byte.end()) {
            out += static_cast<char>(it->second);
        } else {
            // 非 BBPE 映射 codepoint（特殊 token 等）：原样 UTF-8 透传。
            out.append(before, p - before);
        }
    }
    return out;
}

std::string Tokenizer::id_to_piece(uint32_t id, bool skip_special) const {
    // special token：skip 时返回空，否则返回字面量。
    for (const auto& sp : special_tokens_) {
        if (sp.id == id) {
            return skip_special ? std::string() : sp.content;
        }
    }
    if (id < total_size_) {
        return id_to_token_[id];
    }
    return ""; // 未知 id：静默跳过（与 decode 的语义一致）
}

std::string Tokenizer::emit_delta(StreamState& state, const std::vector<uint32_t>& all_ids,
                                  bool skip_special) const {
    // 快路径：无新 token 且无残留字节，直接返回空。
    if (all_ids.size() <= state.sent_count && state.pending_bytes.empty()) {
        return "";
    }
    // Piece 串是 byte-level BPE 的映射码点 UTF-8 编码（见 ByteUnicodeMap），
    // 必须先像 decode() 那样逐 codepoint 还原回原始字节，再交给流式
    // sanitizer——否则多字节字符的组成字节会以映射码点的形式泄漏进输出，
    // 跨 token 拆分的 codepoint 也永远拼不回原文。非映射码点（特殊 token
    // 字面量里的非常规字符）与 decode() 一致按 UTF-8 透传。
    const auto& m = byte_unicode_map();
    std::string raw;
    for (std::size_t k = state.sent_count; k < all_ids.size(); ++k) {
        std::string piece = id_to_piece(all_ids[k], skip_special);
        const char* p = piece.data();
        const char* end = p + piece.size();
        while (p < end) {
            const char* before = p;
            uint32_t cp = decode_utf8(p, end);
            auto it = m.cp_to_byte.find(cp);
            if (it != m.cp_to_byte.end()) {
                raw += static_cast<char>(it->second);
            } else {
                raw.append(before, p - before);
            }
        }
    }
    state.sent_count = all_ids.size();
    return utf8::sanitizeUtf8Streaming(raw, state.pending_bytes);
}

std::string Tokenizer::emit_delta_flush(StreamState& state) const {
    return utf8::sanitizeUtf8Flush(state.pending_bytes);
}

std::string Tokenizer::apply_chat_template(const std::vector<ChatMessage>& messages,
                                           bool add_generation_prompt) const {
    if (chat_template_.empty() || messages.empty()) {
        return "";
    }

    std::string out;

    // 系统提示：messages[0] 若是 system 则取其文本；否则用 default_system_prompt。
    std::string system_prompt;
    bool has_system = false;
    if (messages.front().role == "system") {
        has_system = true;
        for (const auto& c : messages.front().contents) {
            if (c.type == "text") system_prompt += c.content;
        }
    } else if (!chat_template_.default_system_prompt.empty()) {
        has_system = true;
        system_prompt = chat_template_.default_system_prompt;
    }

    if (has_system) {
        auto it = chat_template_.roles.find("system");
        if (it != chat_template_.roles.end()) {
            out += it->second.prefix;
            out += system_prompt;
            out += it->second.suffix;
        } else {
            out += system_prompt;
        }
    }

    // 逐条消息渲染：role prefix + contents + role suffix。
    for (std::size_t mi = 0; mi < messages.size(); ++mi) {
        const auto& msg = messages[mi];
        if (msg.role == "system" && mi == 0) continue; // 系统消息已处理

        auto roleIt = chat_template_.roles.find(msg.role);
        if (roleIt == chat_template_.roles.end()) continue;

        out += roleIt->second.prefix;
        for (const auto& c : msg.contents) {
            if (c.type == "text") {
                out += c.content;
            } else {
                auto ctIt = chat_template_.content_types.find(c.type);
                if (ctIt != chat_template_.content_types.end()) {
                    out += ctIt->second;
                }
            }
        }
        out += roleIt->second.suffix;
    }

    if (add_generation_prompt && !chat_template_.generation_prompt.empty()) {
        out += chat_template_.generation_prompt;
    }

    return out;
}

int32_t Tokenizer::special_token_id(const std::string& literal) const {
    for (const auto& st : special_tokens_) {
        if (st.content == literal) return static_cast<int32_t>(st.id);
    }
    return -1;
}

// 一轮在哪个 token 停：ChatML/Phi 是单个轮末标签，GLM 系的模板根本没有轮末标签
// （一轮靠「下一个角色标签」收尾），generation_config 的 eos_token_id 是三个。
// 未注册进 bin 的字面量永远不命中，所以每张表只对自己那一族生效。
std::vector<int32_t> Tokenizer::stop_token_ids() const {
    std::vector<int32_t> ids;
    for (const char* literal : {kChatMlTurnEnd, kPhiTurnEnd, kGlmEndOfText}) {
        const int32_t id = special_token_id(literal);
        if (id >= 0) ids.push_back(id);
    }
    for (const char* literal : {kGlmUserTag, kGlmObservationTag}) {
        const int32_t id = special_token_id(literal);
        if (id >= 0) ids.push_back(id);
    }
    return ids;
}

// 一 turn 的收尾字面量：渲染「单条 assistant 消息、带引导串」后剥掉前面所有已知
// 部分，剩下的就是这套模板给一轮「封口」的串。ChatML 是轮末标签 + 换行，Phi 是
// <|assistant|> + 换行，GLM 系是空串 —— 那一族的轮边界由下一轮的开头标签承担。
//
// 探针正文用空串而不是 "a"：正文夹在角色前缀和收尾之间，非空正文会被模板的空白处理
// 规则牵连（Phi 的 jinja 在正文为空时会把收尾的最后几个字符吃掉，剥出来的是半截壳）。
// 空正文时剥出来的才是纯收尾；再用「不带引导串」的那次渲染交叉校验，两次剥出来的串
// 必须一致，否则判定这套模板不可用（返回空串，调用方按「无模板」退化）。
std::string Tokenizer::turn_tail_literal() const {
    if (chat_template_.empty()) return "";
    const auto it = chat_template_.roles.find("assistant");
    if (it == chat_template_.roles.end()) return "";
    const std::vector<ChatMessage> msgs = {{"assistant", {{"text", ""}}}};
    const std::string head = it->second.prefix;
    auto peel = [&](bool add_gen) -> std::string {
        const std::string rendered = apply_chat_template(msgs, add_gen);
        std::string base = rendered;
        if (add_gen) {   // 末尾的引导串按长度截掉（它就是这个 prefix）
            if (base.size() < head.size()) return "";
            base.erase(base.size() - head.size());
        }
        // 剩下的形如 [系统段] + 引导串 + 收尾：从后往前剥，先剥掉与引导串等长的那段，
        // 再在其中以 prefix 开头的位置切开。
        if (base.size() < head.size()) return "";
        const std::string tail_of_base = base.substr(base.size() - head.size());
        if (tail_of_base != head) return "";
        const std::string body = base.substr(0, base.size() - head.size());
        const std::size_t at = body.rfind(head);
        if (at == std::string::npos) return "";
        return body.substr(at + head.size());
    };
    const std::string with_gen = peel(/*add_gen=*/true);
    if (with_gen.empty()) return "";
    // 交叉校验：不追加引导串时，收尾就是「剥掉 prefix 之后剩的全部」。
    const std::string plain = apply_chat_template(msgs, false);
    if (plain.size() < head.size() || plain.compare(0, head.size(), head) != 0)
        return "";
    const std::string without_gen = plain.substr(head.size());
    if (without_gen != with_gen) return "";
    return with_gen;
}

int32_t Tokenizer::image_pad_token_id() const {
    return special_token_id(kImagePadTag);
}

} // namespace qwen
