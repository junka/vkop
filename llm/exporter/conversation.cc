// junka @ 2026
// Conversation 实现：每轮的渲染/tokenize/图像占位展开，整段历史的拼接与按预算
// 裁切。设计约束见头文件（assistant 轮不重分词、每轮整段重 prefill）。

#include "conversation.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>

namespace vkop {
namespace export_ {

Conversation::Conversation(const qwen::Tokenizer& tok) : tok_(tok) {
    const int32_t end = tok_.im_end_token_id();
    im_end_id_ = end < 0 ? 0 : static_cast<uint32_t>(end);
    const int32_t pad = tok_.image_pad_token_id();
    image_pad_id_ = pad < 0 ? 0 : static_cast<uint32_t>(pad);

    // 探测模板能不能拆成「前缀 / 后缀」两段：分别渲染空正文和单字符正文的 user
    // 轮，两者之差就是那段正文编码后的样子（占位符会被展开成 N 个重复 token，
    // 所以「前缀+正文+后缀」不能拿整串直接切）。带正文的只用来取后缀 —— 它不含
    // 正文，单独 encode 是精确的；user 和 assistant 在这套模板里共用同一个收尾串。
    const std::string user_prefix = tok_.apply_chat_template(
        {{"user", {{"text", ""}}}}, /*add_generation_prompt=*/false);
    const std::string user_turn =
        tok_.apply_chat_template({{"user", {{"text", "a"}}}}, false);
    // assistant 轮的开头 = 引导串（角色模板的 prefix，不含收尾）。
    auto asst_role = tok_.chat_template().roles.find("assistant");
    const std::string asst_prefix =
        asst_role == tok_.chat_template().roles.end()
            ? std::string() : asst_role->second.prefix;
    if (user_prefix.empty() || user_turn.size() <= user_prefix.size() ||
        asst_prefix.empty() || !im_end_id_) {
        // 旧 tokenizer.bin 没烘 chat template：只能退化成「每轮独立、正文原样
        // encode」，多轮累积交给调用方关掉（has_template() == false）。
        return;
    }
    const std::string suffix_str = user_turn.substr(user_prefix.size());
    user_suffix_ = tok_.encode(suffix_str);
    gen_ids_ = tok_.encode(asst_prefix);
    if (user_suffix_.empty() || gen_ids_.empty()) {
        user_suffix_.clear();
        gen_ids_.clear();
    } else {
        // 引导串不能只 encode 它自己：BBPE 的 ByteLevel 把「词首空格」编成独立
        // 标记，收尾串末尾换行之后的那个空格属于 " assistant" 这个 token，单独
        // encode 会把它丢掉（HF 是整段一次 encode 的）。所以 encode「收尾 +
        // 引导」整串再剥掉与收尾等长的一段。
        const std::vector<uint32_t> seam = tok_.encode(suffix_str + asst_prefix);
        if (seam.size() <= user_suffix_.size() ||
            !std::equal(user_suffix_.begin(), user_suffix_.end(), seam.begin())) {
            // 拼接处被 BPE 合并掉了，剥不出干净的引导段：退回只 encode 引导串
            // 本身（少一个词首空格标记），也不能留下错位的 ids。
            user_suffix_.clear();
            gen_ids_.clear();
        } else {
            gen_ids_.assign(seam.begin() + user_suffix_.size(), seam.end());
        }
    }
}

void Conversation::addUserTurn(const std::string& text, int block_first,
                               int block_count) {
    if (block_count < 0 || block_first < 0 ||
        block_first + block_count > static_cast<int>(blocks_.size())) {
        throw std::runtime_error("Conversation: image block range out of bounds");
    }

    std::vector<qwen::ChatContent> contents;
    contents.reserve(block_count + 1);
    for (int i = 0; i < block_count; ++i) {
        contents.push_back({/*type=*/"image", /*content=*/""});
    }
    contents.push_back({/*type=*/"text", text});

    std::string body = tok_.apply_chat_template({{"user", contents}},
                                               /*add_generation_prompt=*/false);
    // 这个 tokenizer.bin 没烘进 chat template：退回原文（与引入模板之前的驱动
    // 行为一致），但带图输入就没法继续了。
    if (body.empty()) {
        if (block_count > 0) {
            throw std::runtime_error(
                "Conversation: no chat template, cannot render image content");
        }
        body = text;
    }

    const std::vector<uint32_t> raw = tok_.encode(body);
    int placeholders = 0;
    if (image_pad_id_) {
        for (uint32_t id : raw) {
            if (id == image_pad_id_) ++placeholders;
        }
    }
    if (placeholders != block_count) {
        throw std::runtime_error(
            "Conversation: template rendered " + std::to_string(placeholders) +
            " image placeholder(s) for a turn carrying " +
            std::to_string(block_count) + " image(s)");
    }

    // 每个占位展开成该图的 n_img 份（HF processor 按 grid_thw 做同样的事）。
    Turn turn;
    turn.role = "user";
    int next_block = block_first;
    for (uint32_t id : raw) {
        if (image_pad_id_ && id == image_pad_id_) {
            const ImageBlock& blk = blocks_[next_block];
            if (blk.n_img <= 0) {
                throw std::runtime_error("Conversation: image block " +
                                         std::to_string(next_block) +
                                         " has n_img=" + std::to_string(blk.n_img));
            }
            turn.spans.push_back({static_cast<int>(turn.ids.size()), blk.n_img,
                                  next_block});
            for (int k = 0; k < blk.n_img; ++k) turn.ids.push_back(image_pad_id_);
            ++next_block;
        } else {
            turn.ids.push_back(id);
        }
    }
    total_len_ += static_cast<int>(turn.ids.size());
    turns_.push_back(std::move(turn));
}

void Conversation::addAssistantTurn(const std::vector<uint32_t>& gen_ids) {
    Turn turn;
    turn.role = "assistant";
    // 正文 = 生成时的原始 ids，一个不改；外面套上角色前缀和收尾。
    turn.ids = gen_ids_;
    turn.ids.insert(turn.ids.end(), gen_ids.begin(), gen_ids.end());
    if (!has_template()) {                      // 无模板：正文原样，不累积
        total_len_ += static_cast<int>(turn.ids.size());
        turns_.push_back(std::move(turn));
        return;
    }
    if (turn.ids.size() == gen_ids_.size()) {   // 空回复：只补收尾
        turn.ids.insert(turn.ids.end(), user_suffix_.begin(), user_suffix_.end());
    } else if (turn.ids.back() != im_end_id_) {
        // 没生成到结束符（被 max_new 截断）：补完整收尾。
        turn.ids.insert(turn.ids.end(), user_suffix_.begin(), user_suffix_.end());
    } else if (turn.ids.size() <= gen_ids_.size() + 1 ||
               turn.ids[turn.ids.size() - 2] != user_suffix_[0]) {
        // 生成循环在结束符处就停了，收尾串的其余部分（换行）没进 out_ids。补上
        // 结束符之后的那些 —— 少了它们，下一轮序列就不是上一轮的合法扩展，跨轮
        // KV 前缀复用也就无从校验。
        turn.ids.insert(turn.ids.end(), user_suffix_.begin() + 1,
                        user_suffix_.end());
    }
    total_len_ += static_cast<int>(turn.ids.size());
    turns_.push_back(std::move(turn));
}

RenderedContext Conversation::render() const {
    RenderedContext out;
    out.ids.reserve(static_cast<size_t>(total_len_) + gen_ids_.size());
    int cursor = 0;
    for (const auto& turn : turns_) {
        for (const auto& s : turn.spans) {
            out.spans.push_back({cursor + s.start, s.count, s.block});
        }
        out.ids.insert(out.ids.end(), turn.ids.begin(), turn.ids.end());
        cursor += static_cast<int>(turn.ids.size());
    }
    out.ids.insert(out.ids.end(), gen_ids_.begin(), gen_ids_.end());

    out.mm_types.assign(out.ids.size(), 0);
    for (const auto& s : out.spans) {
        for (int i = 0; i < s.count; ++i) out.mm_types[s.start + i] = 1;
    }
    return out;
}

bool Conversation::trimToBudget(int max_tokens) {
    if (max_tokens <= 0) return false;
    bool dropped = false;
    // 一问一答是一对，从最旧的一对开始整对丢；最新一对永不丢。
    while (total_len_ + static_cast<int>(gen_ids_.size()) > max_tokens &&
           turns_.size() > 2) {
        for (int k = 0; k < 2 && turns_.size() > 2; ++k) {
            total_len_ -= static_cast<int>(turns_.front().ids.size());
            turns_.erase(turns_.begin());
            dropped = true;
        }
    }
    return dropped;
}

void Conversation::clear() {
    turns_.clear();
    total_len_ = 0;
}

bool Conversation::prefixMatch(const std::vector<uint32_t>& old_ids,
                               const std::vector<uint32_t>& new_ids) {
    if (new_ids.size() < old_ids.size()) return false;
    for (size_t i = 0; i < old_ids.size(); ++i) {
        if (new_ids[i] != old_ids[i]) return false;
    }
    return true;
}

} // namespace export_
} // namespace vkop
