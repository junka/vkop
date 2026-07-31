// Copyright 2025 @junka
#ifndef OPS_EMBEDDING_FORWARD_HPP_
#define OPS_EMBEDDING_FORWARD_HPP_

#include "ops/BufferBase.hpp"
#include "ops/PimplFacade.hpp"

// Port of cuEmbed's forward propagation path (gather-reduce). Buffer-only
// op: no image path (the gather pattern has no NCHW->RGBA representation).
// fp32, int32 indices this pass.

extern "C" {
extern unsigned char buffer_embedding_forward_spv[];
extern unsigned int buffer_embedding_forward_spv_len;
}

namespace vkop {
namespace ops {

struct alignas(16) EmbeddingPC {
    int embed_width;
    int batch_size;
    int num_hots;     // fixed hotness; 0 if CSR
    int combine_mode; // 0=sum, 1=mean, 2=concat
    int is_csr;       // 1=CSR, 0=fixed
    int is_weighted;  // 1=weighted, 0=unweighted
    int _pad0;
    int _pad1;
};

// combine_mode codes (shared with the shader).
enum class EmbedCombineMode { kSum = 0, kMean = 1, kConcat = 2 };

class EmbeddingForwardBuffer : public BufferFactory {
  public:
    EmbeddingForwardBuffer()
        : BufferFactory(OpType::EMBEDDING_FORWARD, buffer_embedding_forward_spv,
                        buffer_embedding_forward_spv_len,
                        {DESCRIPTOR_TYPE_STORAGE, DESCRIPTOR_TYPE_STORAGE,
                         DESCRIPTOR_TYPE_STORAGE, DESCRIPTOR_TYPE_STORAGE,
                         DESCRIPTOR_TYPE_STORAGE},
                        sizeof(EmbeddingPC)) {}

    void setAttribute(const std::unordered_map<std::string, std::string>
                          &attributes) override {
        if (attributes.find("combine_mode") != attributes.end()) {
            combine_mode_ = std::stol(attributes.at("combine_mode"));
        }
        if (attributes.find("num_hots") != attributes.end()) {
            num_hots_ = std::stol(attributes.at("num_hots"));
        }
        if (attributes.find("is_csr") != attributes.end()) {
            is_csr_ = std::stol(attributes.at("is_csr"));
        }
        if (attributes.find("is_weighted") != attributes.end()) {
            is_weighted_ = std::stol(attributes.at("is_weighted"));
        }
        if (attributes.find("embed_width") != attributes.end()) {
            embed_width_ = std::stol(attributes.at("embed_width"));
        }
    }

  private:
    void execute(
        const std::vector<std::shared_ptr<core::ITensor>> &inputs,
        const std::vector<std::shared_ptr<core::ITensor>> &outputs) override {
        // inputs: [0]=embedding [num_cat, embed_width] float
        //         [1]=indices int32
        //         [2]=offsets int32 (CSR) or nullptr
        //         [3]=weights float (or nullptr)
        auto embed_shape = inputs[0]->getShape();
        int ew = embed_width_ > 0
                     ? embed_width_
                     : (embed_shape.size() >= 2 ? embed_shape[1] : 1);
        int batch_size;
        if (is_csr_ != 0) {
            batch_size = core::as_tensor<int>(inputs[2])->num_elements() - 1;
        } else {
            batch_size =
                core::as_tensor<int>(inputs[1])->num_elements() / num_hots_;
        }

        // output shape: sum/mean -> [batch, ew]; concat -> [batch*hot, ew]
        int out_elems;
        if (combine_mode_ == static_cast<int>(EmbedCombineMode::kConcat)) {
            int hot = is_csr_ != 0 ? 0 : num_hots_; // CSR+concat forbidden
            out_elems = batch_size * hot * ew;
        } else {
            out_elems = batch_size * ew;
        }
        std::vector<int> out_shape{out_elems}; // 1-D flat
        dispatch_by_dtype(outputs[0]->dtype(), [&](auto dummy) {
            using T = decltype(dummy);
            auto output = core::as_tensor<T>(outputs[0]);
            if (output->size() == 0) {
                output->resize(out_shape);
            }
            bind_ssbo<T>(outputs[0], /*is_output=*/true);
        });
        bind_ssbo<float>(inputs[0], /*is_output=*/false); // embedding
        bind_ssbo<int>(inputs[1], /*is_output=*/false);   // indices
        // offsets: bind real buffer if CSR, else dummy
        bool has_offsets = (is_csr_ != 0 && inputs.size() > 2 && inputs[2]);
        if (has_offsets) {
            bind_ssbo<int>(inputs[2], /*is_output=*/false);
        } else {
            objs_.emplace_back(dummy_buffer_);
        }
        // weights: bind real buffer if weighted, else dummy. When offsets
        // are absent the weights slot is inputs[2] (if present), since the
        // op's input layout is [embed, idx, off?, wt?] where off is skipped
        // when not CSR.
        int wt_slot = has_offsets ? 3 : 2;
        bool has_weights =
            (is_weighted_ != 0 && static_cast<int>(inputs.size()) > wt_slot &&
             inputs[wt_slot]);
        if (has_weights) {
            bind_ssbo<float>(inputs[wt_slot], /*is_output=*/false);
        } else {
            objs_.emplace_back(dummy_buffer_);
        }

        EmbeddingPC pc{};
        pc.embed_width = ew;
        pc.batch_size = batch_size;
        pc.num_hots = num_hots_;
        pc.combine_mode = combine_mode_;
        pc.is_csr = is_csr_;
        pc.is_weighted = is_weighted_;
        submit(&pc, UP_DIV(batch_size, 16), UP_DIV(ew, 16), 1);
    }

    int embed_width_ = 0;
    int num_hots_ = 0;
    int combine_mode_ = static_cast<int>(EmbedCombineMode::kSum);
    int is_csr_ = 0;
    int is_weighted_ = 0;
};

// Façade: buffer-only (no image path). backend_buffer is accepted but the
// buffer impl is always used.
class EmbeddingForward : public PimplFacade {
  public:
    EmbeddingForward(int /*fp16*/, bool backend_buffer)
        : PimplFacade(OpType::EMBEDDING_FORWARD) {
        (void)backend_buffer;
        impl_ = std::make_unique<EmbeddingForwardBuffer>();
    }
};

} // namespace ops
} // namespace vkop

#endif // OPS_EMBEDDING_FORWARD_HPP_
