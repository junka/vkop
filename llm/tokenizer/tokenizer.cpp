#include "tokenizer.hpp"
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

} // namespace

Tokenizer::~Tokenizer() {
    if (mmap_data_ && mmap_data_ != MAP_FAILED) {
        munmap(mmap_data_, mmap_size_);
    }
}

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
    vocab_size_ = *reinterpret_cast<const uint32_t*>(ptr); ptr += 4;
    uint32_t merge_count = *reinterpret_cast<const uint32_t*>(ptr); ptr += 4;
    uint32_t special_count = *reinterpret_cast<const uint32_t*>(ptr); ptr += 4;
    ptr += 4; // reserved

    std::cout << "[Tokenizer] Vocab: " << vocab_size_ << ", Merges: " << merge_count
              << ", Special: " << special_count << "\n";

    // 解析 Vocab Section
    id_to_token_.resize(vocab_size_);
    token_to_id_.reserve(vocab_size_);
    for (uint32_t i = 0; i < vocab_size_; ++i) {
        if (ptr + 2 > end) throw std::runtime_error("truncated vocab");
        uint16_t len = *reinterpret_cast<const uint16_t*>(ptr); ptr += 2;
        if (ptr + len + 4 > end) throw std::runtime_error("truncated vocab entry");
        std::string token(reinterpret_cast<const char*>(ptr), len); ptr += len;
        uint32_t tid = *reinterpret_cast<const uint32_t*>(ptr); ptr += 4;
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
        uint32_t left = *reinterpret_cast<const uint32_t*>(ptr); ptr += 4;
        uint32_t right = *reinterpret_cast<const uint32_t*>(ptr); ptr += 4;
        uint32_t merged = *reinterpret_cast<const uint32_t*>(ptr); ptr += 4;
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
        if (ptr + 2 > end) throw std::runtime_error("truncated special tokens");
        uint16_t len = *reinterpret_cast<const uint16_t*>(ptr); ptr += 2;
        if (ptr + len + 4 > end) throw std::runtime_error("truncated special entry");
        std::string content(reinterpret_cast<const char*>(ptr), len); ptr += len;
        uint32_t sid = *reinterpret_cast<const uint32_t*>(ptr); ptr += 4;
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
}

std::vector<uint32_t> Tokenizer::encode(const std::string& text) const {
    const auto& m = byte_unicode_map();
    std::vector<uint32_t> tokens;
    tokens.reserve(text.size());

    // 先扫描输入，遇到特殊 token 字面量（如 <|im_start|>）直接吐对应 id，
    // 该段不进 BBPE；其余子串按字节做 BBPE。这模仿 HF tokenizers 的行为：
    // 特殊 token 在 pre-tokenizer 之前被切出，作为原子单元。
    size_t i = 0;
    const size_t n = text.size();
    while (i < n) {
        // 尝试在当前位置匹配最长特殊 token（special_tokens_ 已按长度降序）。
        bool matched = false;
        for (const auto& sp : special_tokens_) {
            const auto& s = sp.content;
            if (s.size() <= n - i && std::memcmp(text.data() + i, s.data(), s.size()) == 0) {
                tokens.push_back(sp.id);
                i += s.size();
                matched = true;
                break;
            }
        }
        if (matched) continue;

        // 普通路径：把这一字节映射成 BBPE token id，累积成一段后整体做 BPE。
        // 这里按字节逐个推入，下面 bpe_merge 会把它们合并；为保证不同段
        // 之间不会跨段合并，每个普通段单独 BPE 后再追加。
        // 简化：逐字节推入同一段，依赖 merge 表不会跨特殊 token 边界合并
        // （因为特殊 token 的 id 不在任何 merge 规则的 key 里）。
        unsigned char c = static_cast<unsigned char>(text[i]);
        auto it = token_to_id_.find(m.byte_to_str[c]);
        if (it != token_to_id_.end()) {
            tokens.push_back(it->second);
        }
        ++i;
    }

    // 执行 BPE 合并。
    bpe_merge(tokens);
    return tokens;
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

} // namespace qwen