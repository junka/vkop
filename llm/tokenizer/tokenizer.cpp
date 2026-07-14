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

// GPT-2 pre_tokenizer 的缩写匹配 alt1: (?i:'s|'t|'re|'ve|'m|'ll|'d)
// 在 cps[i] == '\'' 时尝试，返回匹配的 codepoint 数（0=不匹配）。
std::size_t match_alt1(const std::vector<uint32_t>& cps, std::size_t i) {
    std::size_t n = cps.size();
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
//   | \p{N}                                    alt3
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
void pre_tokenize(const std::string& s, std::vector<std::string>& out) {
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
            std::size_t m = match_alt1(cps, i);
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

        // alt3: \p{N}（单个数字 codepoint，无前缀吸附）
        if (is_number(cp)) { emit(i, i + 1); i += 1; continue; }

        // alt4: ' ?[^\s\p{L}\p{N}]+ [\r\n]*  （前缀仅字面空格 0x20）
        bool cp_nonsln = !is_ws(cp) && !is_letter(cp) && !is_number(cp);
        if ((cp == 0x20 && i + 1 < n && !is_ws(cps[i + 1]) && !is_letter(cps[i + 1]) && !is_number(cps[i + 1]))
            || cp_nonsln) {
            std::size_t j = (cp == 0x20) ? i + 1 : i;
            while (j < n && !is_ws(cps[j]) && !is_letter(cps[j]) && !is_number(cps[j])) ++j; // [^\s\p{L}\p{N}]+
            while (j < n && (cps[j] == 0x0D || cps[j] == 0x0A)) ++j;                          // [\r\n]*
            emit(i, j); i = j; continue;
        }

        // 以下处理 \s。先取从 i 起的最大 \s-run [i..j)。
        if (is_ws(cp)) {
            std::size_t j = i;
            while (j < n && is_ws(cps[j])) ++j;

            // alt5: \s*[\r\n]+ —— run 内含 CR/NL 时，匹配 [i..最后一个CR/NL+1)
            std::size_t last_crlf = std::size_t(-1);
            for (std::size_t k = i; k < j; ++k) {
                if (cps[k] == 0x0D || cps[k] == 0x0A) last_crlf = k;
            }
            if (last_crlf != std::size_t(-1)) {
                emit(i, last_crlf + 1); i = last_crlf + 1; continue;
            }

            // run 内无 CR/NL：
            // alt6: \s+(?!\S)
            if (j == n) {
                // EOF：吞整个 run
                emit(i, j); i = j; continue;
            }
            // j < n，cps[j] 是 \S。回溯留最后一个 \s 给后续吸附。
            if (j - 1 > i) {
                // run>=2：吞 [i..j-1)，剩 1 个 \s 给 alt2/alt4 吸附或下一轮
                emit(i, j - 1); i = j - 1; continue;
            }
            // run==1：alt6 失败（\s+ 需至少 1 个但不留空给 (?!\S)）。
            // 此处单个 \s 紧跟 \S：若 \S 是字母/标点，alt2/alt4 在下一轮会把它吸附；
            // 若 \S 是数字（alt3 无前缀），则该 \s 需独立成 token —— 由 alt7 吞。
            // 直接走 alt7（下方）。
            emit(i, j); i = j; continue; // alt7：吞当前 run
        }

        // 兜底：单 codepoint（理论上 GPT-2 正则覆盖所有输入，不会到这）
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
    ptr += 4; // reserved

    std::cout << "[Tokenizer] Vocab: " << vocab_size_ << ", Merges: " << merge_count
              << ", Special: " << special_count << "\n";

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
    std::string normalized = normalize_nfc(text);
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
        pre_tokenize(seg, pieces);
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
    // 拼接所有新 token 的 piece 字节，过 UTF-8 sanitizer 保证增量是合法 UTF-8。
    std::string raw;
    for (std::size_t k = state.sent_count; k < all_ids.size(); ++k) {
        raw.append(id_to_piece(all_ids[k], skip_special));
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

} // namespace qwen
