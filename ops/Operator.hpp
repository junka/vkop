// Copyright 2025 @junka
#ifndef OPS_OPERATOR_HPP_
#define OPS_OPERATOR_HPP_

#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "core/Tensor.hpp"
#include "ops/Ops.hpp"
#include "vulkan/VulkanCommandPool.hpp"
#include "vulkan/VulkanDevice.hpp"

#define UP_DIV(x, y) (((x) + (y)-1) / (y))

namespace vkop {
namespace ops {

class Operator {
  public:
    explicit Operator(OpType type, uint8_t *spv, uint32_t spv_len,
                      size_t pc_size = 0)
        : type_(type), pc_size_(pc_size), spv_(spv), spv_len_(spv_len) {}
    virtual ~Operator() {
        m_cmdpool_ = nullptr;
        m_dev_ = nullptr;
    };
    Operator(const Operator &) = delete;
    Operator &operator=(const Operator &) = delete;
    Operator(Operator &&) = delete;
    Operator &operator=(Operator &&) = delete;

    virtual void set_runtime_device(std::shared_ptr<VulkanDevice> &dev,
                                    std::shared_ptr<VulkanCommandPool> &cmdpool,
                                    std::shared_ptr<VulkanCommandBuffer> &cmd) {
        m_dev_ = dev;
        m_cmdpool_ = cmdpool;
        // m_cmd_ = std::make_shared<VulkanCommandBuffer>(m_dev_, m_cmdpool_);
        m_cmd_ = cmd;
        create_pipeline();
    }
    virtual void create_pipeline() {
        VkDevice device = m_dev_->getLogicalDevice();
        pipeline_ = std::make_shared<VulkanPipeline>(
            device, types_, pc_size_, reinterpret_cast<const uint32_t *>(spv_),
            spv_len_);
    }

    virtual void setAttribute(
        const std::unordered_map<std::string, std::string> &attributes) {
        if (!attributes.empty()) {
            // for (const auto &attr : attributes) {
            //     std::cout << "Warning: Unused attribute " << attr.first
            //               << " in operator." << std::endl;
            // }
        }
    }
    virtual inline std::vector<int> parse_attr_list(const std::string &str) {
        std::vector<int> result;
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
                result.emplace_back(std::stoi(item));
            }
        }
        return result;
    }

    virtual OpType get_type() { return type_; }

    void onExecute(const std::vector<std::shared_ptr<core::ITensor>> &inputs,
                   const std::vector<std::shared_ptr<core::ITensor>> &outputs) {

        // m_cmd_->wait();
        // m_cmd_->begin();
        execute(inputs, outputs);
    }

  protected:
    std::shared_ptr<VulkanDevice> m_dev_;
    std::shared_ptr<VulkanCommandPool> m_cmdpool_;
    std::shared_ptr<VulkanCommandBuffer> m_cmd_;
    std::shared_ptr<VulkanPipeline> pipeline_;
    OpType type_;
    size_t pc_size_ = 0;
    uint8_t *spv_;
    uint32_t spv_len_;
    int n_imgs_ = 0;

    // we should release objs_ here, since for some intermediate tensor, we will
    // release them in the end of the execution.
    std::vector<std::shared_ptr<VulkanResource>> objs_;
    std::vector<VkDescriptorType> types_;

    using SupportedTypes = std::tuple<float, uint16_t>;
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
            throw std::runtime_error(
                "Unsupported dtype: " + std::string(dtype.name()) +
                ". Supported: float, uint16_t (fp16)");
        }
    }

    virtual void submit(void *ptr, int out_width, int out_height) {

        pipeline_->updateDescriptorSets(objs_, n_imgs_);
#ifdef USE_MEASURE_TIME
        VkDevice device = m_dev_->getLogicalDevice();
        VulkanQueryPool query_pool(device, 2, VK_QUERY_TYPE_TIMESTAMP);
#endif
        m_cmd_->bind(*pipeline_);
#ifdef USE_MEASURE_TIME
        query_pool.begin(m_cmd_->get());
#endif
        if (ptr) {
            m_cmd_->push_constants(*pipeline_, pc_size_, ptr);
        }
        m_cmd_->dispatch(out_width, out_height);
#ifdef USE_MEASURE_TIME
        query_pool.end(m_cmd_->get());
#endif
        // m_cmd_->end();
        // m_cmd_->submit(m_dev_->getComputeQueue());
#ifdef USE_MEASURE_TIME
        auto r = query_pool.getResults();
        LOG_INFO("Time: %f s", static_cast<double>(r[1] - r[0]) * (1e-9) *
                                   m_dev_->getTimestampPeriod());
#endif

        objs_.clear();
    }

  private:
    virtual void
    execute(const std::vector<std::shared_ptr<core::ITensor>> &inputs,
            const std::vector<std::shared_ptr<core::ITensor>> &outputs) = 0;
};

} // namespace ops
} // namespace vkop
#endif // OPS_OPERATOR_HPP_
