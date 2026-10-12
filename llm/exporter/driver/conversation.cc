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
    for (const int32_t id : tok_.stop_token_ids()) {
        stop_ids_.push_back(static_cast<uint32_t>(id));
    }
    const int32_t pad = tok_.image_pad_token_id();
    image_pad_id_ = pad < 0 ? 0 : static_cast<uint32_t>(pad);

    // assistant 引导串 = 角色模板的 prefix；一 turn 的收尾串 = 同一套模板给一轮
    // 封口的字面量（GLM 系为空串）。收尾串单独 encode 是精确的。
    auto asst_role = tok_.chat_template().roles.find("assistant");
    const std::string asst_prefix =
        asst_role == tok_.chat_template().roles.end()
            ? std::string() : asst_role->second.prefix;
    const std::string tail_str = tok_.turn_tail_literal();
    turn_tail_ = tok_.encode(tail_str);
    if (asst_prefix.empty() || stop_ids_.empty() ||
        (!tail_str.empty() && turn_tail_.empty())) {
        // 退化成「每轮独立、正文原样 encode」，多轮累积交给调用方关掉
        // （has_template() == false）：旧 tokenizer.bin 没烘模板，或者词表里根本没有
        // 停止符，或者收尾字面量剥不干净（encode 出空 ids）—— 三种都没法可靠拼历史。
        return;
    }
    gen_ids_ = tok_.encode(asst_prefix);
    if (gen_ids_.empty()) return;
    // 引导串不能只 encode 它自己：BBPE 的 ByteLevel 把「词首空格」编成独立
    // 标记，收尾串末尾换行之后的那个空格属于 " assistant" 这个 token，单独
    // encode 会把它丢掉（HF 是整段一次 encode 的）。所以 encode「收尾 +
    // 引导」整串再剥掉与收尾等长的一段。收尾串为空时这条退化成 encode(引导串)
    // 本身，剥不出错的东西。
    const std::vector<uint32_t> seam = tok_.encode(tail_str + asst_prefix);
    if (seam.size() <= turn_tail_.size() ||
        !std::equal(turn_tail_.begin(), turn_tail_.end(), seam.begin())) {
        // 拼接处被 BPE 合并掉了，剥不出干净的引导段：保留上面单独 encode 的
        // 引导串（少一个词首空格标记），不能留下错位的 ids。
        return;
    }
    gen_ids_.assign(seam.begin() + turn_tail_.size(), seam.end());
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
    // 正文里带着的那个停止符要不要留在本轮序列里，取决于它在模板里的身份：
    //  - ChatML/Phi：<|im_end|> / <|end|> 就是「本轮的收尾」，模板自己也是这么写的
    //    （收尾串里含它），留着才对。
    //  - GLM 系：模板没有轮末标签，模型生成出来的 <|user|> 是**下一轮的开头**，下一轮
    //    的 apply_chat_template 还会再渲染一次 —— 留在这里就成了双份标签，序列
    //    不再是 HF 渲染出来的那个串。收尾串为空正是这一族的信号。
    // 截断到正文一个 token 都没生成时，别把前缀尾部那个 token 判成「带收尾」。
    std::size_t keep = gen_ids.size();
    if (turn_tail_.empty() && keep && isStop(gen_ids.back())) --keep;

    Turn turn;
    turn.role = "assistant";
    // 正文 = 生成时的原始 ids，一个不改；外面套上角色前缀和收尾。
    turn.ids = gen_ids_;
    turn.ids.insert(turn.ids.end(), gen_ids.begin(),
                    gen_ids.begin() + static_cast<std::ptrdiff_t>(keep));
    if (!has_template()) {                      // 无模板：正文原样，不累积
        total_len_ += static_cast<int>(turn.ids.size());
        turns_.push_back(std::move(turn));
        return;
    }
    if (keep == gen_ids.size()) {
        // 没在本轮正文里停下来（被 max_new 截断，或压根没生成正文）：补齐收尾串。
        // GLM 系的收尾串是空串，这条什么都不做 —— 下一轮的开头标签自会封口。
        turn.ids.insert(turn.ids.end(), turn_tail_.begin(), turn_tail_.end());
    } else if (turn_tail_.size() > 1 &&
               turn.ids.size() >= gen_ids_.size() + turn_tail_.size() &&
               turn.ids[turn.ids.size() - turn_tail_.size()] == turn_tail_[0]) {
        // 生成循环在停止符处就停了，收尾串的**其余**部分（""" + "<|im_end|>" + """ 之后那个换行）没进
        // 正文。少了它们，下一轮序列就不是上一轮的合法扩展，跨轮 KV 前缀复用
        // 也就无从校验。
        turn.ids.insert(turn.ids.end(), turn_tail_.begin() + 1,
                        turn_tail_.end());
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
