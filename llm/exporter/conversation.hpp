// junka @ 2026
// Conversation —— 多轮对话的上下文管理（纯 host 侧，不碰 Runtime / 算子）。
//
// 存的是 token 级的历史：每轮渲染并 tokenize 后的 ids（user 轮含 role 前后缀和
// 已展开的 image pad；assistant 轮含 role 前缀 + 生成时的原始 ids + 结束符后缀），
// 外加每轮里每张图落在序列哪一段。render() 把保留的历史拼成一次 prefill 需要的
// 完整序列。
//
// 两个不显然但决定正确性的点：
//  1. assistant 轮绝不重新分词。BBPE 的 decode→encode 不保证可逆（piece 的词首
//     空格标记在 decode 时被丢掉），把生成文本再 encode 一遍会得到和模型实际
//     「写下」的上下文不一致的 ids，下一轮 prefill 就不是复现而是重写历史。
//     所以回复正文原样保存，只有角色壳（prefix+suffix）走 encode。
//  2. 每轮整段重新 prefill（KV 从零开始），语义与 HF 的 chat template + 全量
//     forward 一致。跨轮续用 KV 是纯加速层，前提是新一轮的 ids 必须是上一轮
//     序列的严格前缀 —— 校验点留在 prefixMatch()，没有它就不能复用（vkop 目前
//     没有 paged KV / block table，也就没有 vLLM 那种 prefix caching）。
//
// 用法见 llm_chat.cpp。

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "tokenizer.hpp"

namespace vkop {
namespace export_ {

// 一张已编码图片的元信息。特征本身留在调用方（llm_chat 的 VisualFeatures），
// Conversation 只需要知道「这张图在 LLM 序列里占多少 token、grid 多少」，
// 因此不依赖任何 GPU/视觉类型。
struct ImageBlock {
    int n_img = 0;        // 展开后占的 token 数（224x224 → 49）
    int grid_t = 0;
    int grid_h = 0;
    int grid_w = 0;
};

// 一次 prefill 的全部序列信息：图像占位已展开，可直接据此建输入。
struct RenderedContext {
    std::vector<uint32_t> ids;
    std::vector<int32_t> mm_types;   // 与 ids 等长：0=文本 1=图片（HF mm_token_type_ids）
    struct Span {
        int start = 0;   // ids 里的位置
        int count = 0;   // 该图占的 token 数
        int block = 0;   // 第几张登记过的图（== 视觉特征块下标）
    };
    // 按序列里出现的顺序 —— 也就是特征拼进 image_features / deepstack_embeds_*
    // 的行顺序。
    std::vector<Span> spans;

    int size() const { return static_cast<int>(ids.size()); }
    int image_tokens() const {
        int n = 0;
        for (const auto& s : spans) n += s.count;
        return n;
    }
};

class Conversation {
public:
    struct Turn {
        std::string role;                        // "user" / "assistant"
        std::vector<uint32_t> ids;               // 整轮 token，image pad 已展开
        std::vector<RenderedContext::Span> spans;  // 本轮每张图（start 相对本轮）
    };

    // 结束符 / 图像 pad 两个特殊 token 的 id 都从 tokenizer 注册表按字面量查，
    // 不在调用方写死数字（换 checkpoint 即失效）。
    explicit Conversation(const qwen::Tokenizer& tok);

    // 图片块在 REPL 之前登记完（视觉塔先跑），之后只读。
    void addBlock(const ImageBlock& b) { blocks_.push_back(b); }
    const std::vector<ImageBlock>& blocks() const { return blocks_; }

    // 一 turn 的图必须是 blocks_ 里连续的 [first, first+count)：图的特征只能按
    // 行拼接进单个输入（一次 memcpy），不支持乱序 gather。CLI 的 --image 顺序
    // 正好满足这个前提。
    void addUserTurn(const std::string& text, int block_first = 0,
                     int block_count = 0);
    // 模型本轮原样生成的 token（含尾部结束符），不重新分词；缺结束符时补齐。
    void addAssistantTurn(const std::vector<uint32_t>& gen_ids);

    // 整段历史 + assistant 引导串。每次都全量重渲染，所以裁掉任何 turn 都不可能
    // 留下半更新的序列。
    RenderedContext render() const;

    // 超出 token 预算时整对（user+assistant）丢最旧的，最新一对永不丢（空
    // prompt 没法回答）。返回是否发生了裁切。max_tokens <= 0 表示不裁。
    bool trimToBudget(int max_tokens);

    // 历史本身的序列长度（不含引导串）：预算用量的正确口径。
    int context_len() const { return total_len_; }
    std::size_t size() const { return turns_.size(); }
    const std::vector<Turn>& turns() const { return turns_; }
    uint32_t image_pad_id() const { return image_pad_id_; }
    uint32_t im_end_id() const { return im_end_id_; }
    // tokenizer.bin 里没烘 chat template（旧产物）时为 false：只能每轮独立跑，
    // 累积历史没有意义（没模板就没法把一轮包成模型看得懂的对话格式）。
    bool has_template() const { return !gen_ids_.empty(); }
    void clear();

    // 跨轮 KV 复用的前提：new_ids 以 old_ids 为严格前缀。目前无调用方，接上
    // 续用 KV 时才用。
    static bool prefixMatch(const std::vector<uint32_t>& old_ids,
                            const std::vector<uint32_t>& new_ids);

private:
    const qwen::Tokenizer& tok_;
    std::vector<ImageBlock> blocks_;
    std::vector<Turn> turns_;
    std::vector<uint32_t> gen_ids_;     // assistant 引导串（角色前缀）
    std::vector<uint32_t> user_suffix_; // 一 turn 的收尾（结束符 + 换行）
    uint32_t image_pad_id_ = 0;
    uint32_t im_end_id_ = 0;
    int total_len_ = 0;                 // Σ turn.ids.size()
};

} // namespace export_
} // namespace vkop
