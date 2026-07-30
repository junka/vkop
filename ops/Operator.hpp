// Copyright 2025 @junka
#ifndef OPS_OPERATOR_HPP_
#define OPS_OPERATOR_HPP_

#include <cassert>
#include <cstddef>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/Tensor.hpp"
#include "ops/Ops.hpp"

namespace vkop {
namespace ops {

class Operator {
  public:
    explicit Operator(OpType type, uint8_t *spv, uint32_t spv_len,
                      const std::vector<VkDescriptorType> &types,
                      size_t pc_size = 0, int fp16 = 0)
        : type_(type), pc_size_(pc_size), spv_(spv), spv_len_(spv_len),
          types_(std::move(types)), fp16_(fp16) {
        assert(pc_size_ <= 128);
        instance_count_.fetch_add(1);

        objs_.reserve(types_.size());
        writes_.resize(types_.size());
    }

    // Pin the compute pipeline's subgroup size. Set to the device's reported
    // subgroup size before set_runtime_device() for shaders that hard-assume a
    // fixed numSubgroups (e.g. softmax.comp's cross-subgroup reduce).
    virtual void set_required_subgroup_size(uint32_t size) {
        required_subgroup_size_ = size;
    }

    virtual ~Operator() {
        for (auto &m_d : m_ds_) {
            if (m_d)
                pipeline_->freeDescriptorSets(m_d);
        }
        auto cnt = instance_count_.fetch_sub(1);
        if (cnt == 1 && dummy_buffer_ != nullptr &&
            dummy_bufferview_ != nullptr) {
            dummy_bufferview_.reset();
            dummy_buffer_.reset();
            dummy_buffer_ = nullptr;
            dummy_bufferview_ = nullptr;
        }
        m_cmdpool_ = nullptr;
        m_dev_ = nullptr;
    };
    Operator(const Operator &) = delete;
    Operator &operator=(const Operator &) = delete;
    Operator(Operator &&) = delete;
    Operator &operator=(Operator &&) = delete;

    virtual void
    set_runtime_device(const std::shared_ptr<VulkanDevice> &dev,
                       const std::shared_ptr<VulkanCommandPool> &cmdpool) {
        m_dev_ = dev;
        m_cmdpool_ = cmdpool;
        if (spv_len_ > 0 && spv_) {
            bool use_uab = update_after_bind_ &&
                           m_dev_->is_support_descriptor_update_after_bind();
            pipeline_ = std::make_unique<VulkanPipeline>(
                m_dev_->getLogicalDevice(), types_, pc_size_,
                reinterpret_cast<const uint32_t *>(spv_), spv_len_, use_uab,
                required_subgroup_size_);
            for (auto &ds : m_ds_) {
                ds = pipeline_->allocDescriptorSets();
            }
        }
        if (!dummy_buffer_) {
            dummy_buffer_ = std::make_shared<VulkanBuffer>(
                m_dev_, 16, UNIFORM | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            assert(dummy_buffer_);
            dummy_bufferview_ = std::make_shared<VulkanBufferView>(
                m_dev_, dummy_buffer_, VK_FORMAT_R32G32B32A32_SFLOAT, 16, 0);
            assert(dummy_bufferview_);
        }
    }

    virtual void setAttribute(
        const std::unordered_map<std::string, std::string> &attributes) {
        if (!attributes.empty()) {
            for (const auto &attr : attributes) {
                std::cout << "attribute " << attr.first << ": " << attr.second
                          << " in operator." << std::endl;
            }
        }
    }
    template <typename T>
    std::vector<T> parse_attr_list(const std::string &str) {
        std::vector<T> result;
        if (str.front() == '[' && str.back() == ']') {
            std::string content = str.substr(1, str.size() - 2);
            size_t comma_count = 0;
            for (char c : content) {
                if (c == ',')
                    ++comma_count;
            }
            size_t estimated_size = comma_count + 1;
            result.reserve(estimated_size);
            std::stringstream ss(content);
            std::string item;
            while (std::getline(ss, item, ',')) {
                try {
                    if (std::is_same<T, float>::value) {
                        result.emplace_back(std::stof(item));
                    } else {
                        result.emplace_back(std::stol(item));
                    }
                } catch (const std::invalid_argument &e) {
                    throw std::runtime_error(
                        "Invalid number in attribute list: " + item);
                } catch (const std::out_of_range &e) {
                    throw std::runtime_error(
                        "Number out of range in attribute list: " + item);
                }
            }
        }
        return result;
    }

    virtual OpType get_type() { return type_; }

    // These members are the public API the runtime/tests drive. They are
    // virtual so a PIMPL façade op can forward each to its image/buffer
    // impl (which owns the real pipeline/command-buffer state).
    virtual std::shared_ptr<VulkanCommandBuffer> get_record() { return m_cmd_; }

    virtual void
    onExecute(const std::vector<std::shared_ptr<core::ITensor>> &inputs,
              const std::vector<std::shared_ptr<core::ITensor>> &outputs,
              int id) {
        if (!m_cmd_) {
            m_cmd_ = std::make_shared<VulkanCommandBuffer>(m_cmdpool_, id);
        }
        m_id_ = id;
        objs_.clear();
        m_cmd_->begin();
        execute(inputs, outputs);
        m_cmd_->end();

        if (trace_) {
            m_cmd_->wait();
            m_dev_->wait_all_done();
            printf("%s: %s\n", convert_optype_to_string(type_).c_str(),
                   name_.c_str());
            int i = 0;
            for (auto input : inputs) {
                printf("input %d:\n", i);
                if (input->dtype() == typeid(int64_t)) {
                    continue;
                }
                dispatch_by_dtype(input->dtype(), [&](auto dummy) {
                    using T = decltype(dummy);
                    auto in = core::as_tensor<T>(input);
                    in->copyToCPU(m_cmdpool_);
                    in->print_tensor();
                    in->toGPU();
                });
                i++;
            }
            printf("output 0:\n");
            dispatch_by_dtype(outputs[0]->dtype(), [&](auto dummy) {
                using T = decltype(dummy);
                auto output = core::as_tensor<T>(outputs[0]);
                output->copyToCPU(m_cmdpool_);
                output->print_tensor();
                output->toGPU();
            });
        }
    }
    virtual void enable_trace() { trace_ = true; }
    virtual void disable_trace() { trace_ = false; }

    virtual void set_name(const std::string &name) { name_ = name; }
    virtual std::string get_name() const { return name_; }

    static std::shared_ptr<VulkanBuffer> dummy_buffer_;
    static std::shared_ptr<VulkanBufferView> dummy_bufferview_;
    static std::atomic<int> instance_count_;

  protected:
    std::shared_ptr<VulkanDevice> m_dev_;
    std::shared_ptr<VulkanCommandPool> m_cmdpool_;
    std::shared_ptr<VulkanCommandBuffer> m_cmd_ = nullptr;

    std::unique_ptr<VulkanPipeline> pipeline_;
    VkDescriptorSet m_ds_[vkop::kInflight] = {nullptr};
    std::vector<VkWriteDescriptorSet> writes_;
    std::vector<VkDescriptorBufferInfo> buffer_infos_;
    std::vector<VkDescriptorImageInfo> image_infos_;
    int m_id_;
    OpType type_;
    size_t pc_size_ = 0;
    uint8_t *spv_ = nullptr;
    uint32_t spv_len_ = 0;
    bool update_after_bind_ = false;
    bool trace_ = false;
    std::string name_;
    uint32_t required_subgroup_size_ = 0;

    // we should release objs_ here, since for some intermediate tensor, we will
    // release them in the end of the execution.
    std::vector<std::shared_ptr<VulkanResource>> objs_;
    std::vector<VkDescriptorType> types_;
    int fp16_ = 0; // 0: fp32, 1: fp16

    using SupportedTypes = std::tuple<float, uint16_t, int, int64_t, int8_t>;
    template <typename Func>
    void dispatch_by_dtype(const std::type_info &dtype, Func &&func) {
        bool dispatched = false;

        std::apply(
            [&](auto... types) {
                (([&] {
                     using T = decltype(types);
                     if (dtype == typeid(T)) {
                         func.template operator()<T>(T{});
                         dispatched = true;
                     }
                 }()),
                 ...);
            },
            SupportedTypes{});

        if (!dispatched) {
            throw std::runtime_error("Unsupported dtype: " +
                                     std::string(dtype.name()));
        }
    }

    virtual void submit(void *ptr, int width, int height, int layers) {
        if (!m_ds_[m_id_]) {
            m_ds_[m_id_] = pipeline_->allocDescriptorSets();
        }
        fillWriteDescriptorSets(m_ds_[m_id_]);
        pipeline_->updateDescriptorSets(writes_);

        m_cmd_->bind(*pipeline_, m_ds_[m_id_]);
        if (ptr) {
            m_cmd_->push_constants(*pipeline_, static_cast<uint32_t>(pc_size_),
                                   ptr);
        }
        m_cmd_->dispatch(width, height, layers);
    }

    // Submit with a dedicated descriptor set for multi-pass operators.
    // Each pass must use its own descriptor set because vkUpdateDescriptorSets
    // modifies the set immediately — all dispatches referencing the same set
    // see the LAST update's bindings, causing earlier passes to read wrong
    // buffers on some drivers (e.g. Intel ANV).
    void submit_per_ds(VkDescriptorSet ds, void *ptr, int width, int height,
                       int layers) {
        fillWriteDescriptorSets(ds);
        pipeline_->updateDescriptorSets(writes_);

        m_cmd_->bind(*pipeline_, ds);
        if (ptr) {
            m_cmd_->push_constants(*pipeline_, static_cast<uint32_t>(pc_size_),
                                   ptr);
        }
        m_cmd_->dispatch(width, height, layers);
    }

    // Allocate a fresh descriptor set from the pipeline's pool.
    VkDescriptorSet allocPassDescriptorSet() {
        return pipeline_->allocDescriptorSets();
    }

    // Return a per-pass descriptor set to the pool.
    void freePassDescriptorSet(VkDescriptorSet ds) {
        pipeline_->freeDescriptorSets(ds);
    }

  private:
    virtual void fillWriteDescriptorSets(VkDescriptorSet ds) {
        for (size_t i = 0; i < types_.size(); i++) {
            writes_[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes_[i].dstSet = ds;
            writes_[i].dstBinding = static_cast<uint32_t>(i);
            writes_[i].dstArrayElement = 0;
            writes_[i].descriptorCount = 1;
            writes_[i].descriptorType = types_[i];
            switch (objs_[i]->getResourceType()) {
            case ResourceType::VK_BUFFER_VIEW:
                writes_[i].pTexelBufferView =
                    std::get<VkBufferView *>(objs_[i]->getDescriptorInfo());
                break;
            case ResourceType::VK_BUFFER:
                writes_[i].pBufferInfo = std::get<VkDescriptorBufferInfo *>(
                    objs_[i]->getDescriptorInfo());
                break;
            case ResourceType::VK_IMAGE:
                writes_[i].pImageInfo = std::get<VkDescriptorImageInfo *>(
                    objs_[i]->getDescriptorInfo());
                break;
            default:
                break;
            }
        }
    }

    virtual void
    execute(const std::vector<std::shared_ptr<core::ITensor>> &inputs,
            const std::vector<std::shared_ptr<core::ITensor>> &outputs) = 0;
};

} // namespace ops
} // namespace vkop
#endif // OPS_OPERATOR_HPP_
