#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <stdexcept>

#include <re2/re2.h>

namespace qwen {

// 用于哈希 std::pair 的自定义结构体
struct PairHash {
    template <class T1, class T2>
    std::size_t operator() (const std::pair<T1, T2>& p) const {
        auto h1 = std::hash<T1>{}(p.first);
        auto h2 = std::hash<T2>{}(p.second);
        return h1 ^ (h2 << 1);
    }
};

class Tokenizer {
public:
    Tokenizer() = default;
    ~Tokenizer();

    // 禁止拷贝
    Tokenizer(const Tokenizer&) = delete;
    Tokenizer& operator=(const Tokenizer&) = delete;

    // 加载二进制模型
    void load(const std::string& bin_path);

    // 核心分词接口 (Encode)
    std::vector<uint32_t> encode(const std::string& text) const;

    // 核心解码接口 (Decode)
    std::string decode(const std::vector<uint32_t>& ids) const;

private:
    // BBPE 核心合并逻辑
    void bpe_merge(std::vector<uint32_t>& tokens) const;

    // 把一段 pre-tokenizer 切出的 span（已是字节序列）做字节→BBPE id + BPE 合并。
    void encode_segment(const std::string& seg, std::vector<uint32_t>& out) const;

    // mmap 相关成员
    void* mmap_data_ = nullptr;
    size_t mmap_size_ = 0;

    // 词表相关
    uint32_t vocab_size_ = 0;          // model.vocab 的条目数（不含 added_tokens）
    uint32_t total_size_ = 0;          // vocab_size_ + special tokens，id_to_token_ 的实际容量
    std::vector<std::string> id_to_token_;
    std::unordered_map<std::string, uint32_t> token_to_id_;

    // 合并规则相关 (核心哈希表)
    std::unordered_map<std::pair<uint32_t, uint32_t>, uint32_t, PairHash> merge_map_;
    // (left_id,right_id) -> BPE 优先级 rank（merge 在 bin 文件中的顺序）。
    // bpe_merge 每轮选 rank 最小的相邻 pair 合并，是标准 GPT-2 BPE 行为。
    std::unordered_map<std::pair<uint32_t, uint32_t>, uint32_t, PairHash> merge_rank_;

    // pre_tokenizer 的 GPT-2 正则。Qwen3-VL 原版含 \s+(?!\S) 负向前瞻，re2 不支持，
    // 故去掉该子句（保留兜底的 \s+）；对常规输入的切分结果与 HF 一致。
    // 模式整体包一层捕获组，用 RE2::Match 逐段推进。
    std::unique_ptr<RE2> pre_regex_;

    // 特殊 token（如 <|im_start|>）。encode 时先做最长字面量匹配，命中即输出
    // 单个 id，跳过该段；该段不再进 BBPE。按长度降序排列以保证最长匹配优先。
    struct SpecialToken {
        std::string content;
        uint32_t id;
    };
    std::vector<SpecialToken> special_tokens_; // 按长度降序

};

} // namespace qwen