#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <stdexcept>

#include <utf8proc.h>

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

// 流式 decode 的逐 slot 状态。sent_count 是已喂给 emit_delta 的 token 数，
// pending_bytes 是跨调用携带的尾部不完整 UTF-8 字节（多字节 codepoint 被拆到
// 多个 token 时，前几次调用会暂存尾部、不输出半字符）。
struct StreamState {
    std::size_t sent_count = 0;
    std::string pending_bytes;
};

// Chat template 的单个角色配置（如 system/user/assistant 各自的 prefix/suffix）。
struct ChatRole {
    std::string prefix;
    std::string suffix;
};

// 从 qwen3_vl.bin 的 ChatTemplate Section 加载出的模板数据。roles 是
// role 名->(prefix,suffix)；content_types 是非文本内容（image/video）的占位
// 格式串；generation_prompt 是 add_generation_prompt 时追加的引导串。
struct ChatTemplate {
    std::unordered_map<std::string, ChatRole> roles;
    std::unordered_map<std::string, std::string> content_types; // type -> format
    std::string generation_prompt;
    std::string default_system_prompt;

    bool empty() const noexcept { return roles.empty(); }
};

// 一条对话消息的内容项。type 为 "text" 时用 content 原文；为 "image"/"video"
// 等时忽略 content，改用 ChatTemplate.content_types[type] 的占位串。
struct ChatContent {
    std::string type;    // "text" / "image" / "video"
    std::string content; // type=="text" 时为文本，否则可空
};

struct ChatMessage {
    std::string role;    // "system" / "user" / "assistant"
    std::vector<ChatContent> contents;
};

class Tokenizer {
public:
    Tokenizer() = default;
    ~Tokenizer();

    // 禁止拷贝
    Tokenizer(const Tokenizer&) = delete;
    Tokenizer& operator=(const Tokenizer&) = delete;

    // 加载二进制模型（含 vocab/merges/special/chat-template 四段）。
    void load(const std::string& bin_path);

    // 核心分词接口 (Encode)
    std::vector<uint32_t> encode(const std::string& text) const;

    // 核心解码接口 (Decode) —— 单次全量。
    std::string decode(const std::vector<uint32_t>& ids) const;

    // 单 token 反查：返回该 id 对应的原始 piece 字节串。special token 在
    // skip_special=true 时返回 ""；未知 id 返回 ""。不做 UTF-8 合法化（流式
    // 路径由 emit_delta 统一过 sanitizer）。
    std::string id_to_piece(uint32_t id, bool skip_special = true) const;

    // 流式增量解码：消费 all_ids[state.sent_count..end) 各 token 的 piece，
    // 过 UTF-8 sanitizer 后返回 well-formed UTF-8 增量。state.sent_count 前进
    // 到 all_ids.size()；尾部不完整 codepoint 存入 state.pending_bytes。
    std::string emit_delta(StreamState& state, const std::vector<uint32_t>& all_ids,
                           bool skip_special = true) const;

    // 流式终止：把 state.pending_bytes 里残留字节各转成 U+FFFD 并清空。合法
    // 流上返回空串。
    std::string emit_delta_flush(StreamState& state) const;

    // Chat template 访问。bin 未含 ChatTemplate Section 时返回空模板。
    const ChatTemplate& chat_template() const noexcept { return chat_template_; }

    // 渲染对话为 prompt 字符串（调用方再 encode）。messages[0] 若是 system
    // 则用作系统提示，否则用 default_system_prompt（若有）。add_generation_prompt
    // 时末尾追加 generation_prompt。返回空串表示模板未加载或无消息。
    std::string apply_chat_template(const std::vector<ChatMessage>& messages,
                                    bool add_generation_prompt = true) const;

private:
    // BBpe 核心合并逻辑
    void bpe_merge(std::vector<uint32_t>& tokens) const;

    // 把一段 pre-tokenizer 切出的 span（已是字节序列）做字节→BBPE id + BPE 合并。
    void encode_segment(const std::string& seg, std::vector<uint32_t>& out) const;

    std::string normalize_nfc(const std::string& text) const;

    void post_process(std::vector<uint32_t>& ids) const;

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

    // 特殊 token（如 <|im_start|>）。encode 时先做最长字面量匹配，命中即输出
    // 单个 id，跳过该段；该段不再进 BBPE。按长度降序排列以保证最长匹配优先。
    struct SpecialToken {
        std::string content;
        uint32_t id;
    };
    std::vector<SpecialToken> special_tokens_; // 按长度降序

    // Chat template（从 bin 的 ChatTemplate Section 加载；旧 bin 无此段则为空）。
    ChatTemplate chat_template_;
};

} // namespace qwen
