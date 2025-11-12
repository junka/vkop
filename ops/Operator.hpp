// Copyright 2025 @junka
#ifndef OPS_OPERATOR_HPP_
#define OPS_OPERATOR_HPP_

#include <sstream>
#include <string>
#include <unordered_map>
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
    explicit Operator(OpType type) : type_(type) {}
    virtual ~Operator() {
        m_cmdpool_ = nullptr;
        m_dev_ = nullptr;
    };
    Operator(const Operator &) = delete;
    Operator &operator=(const Operator &) = delete;
    Operator(Operator &&) = delete;
    Operator &operator=(Operator &&) = delete;

    virtual void
    set_runtime_device(VkPhysicalDevice phydev,
                       std::shared_ptr<VulkanDevice> dev,
                       std::shared_ptr<VulkanCommandPool> cmdpool) {
        m_phydev_ = phydev;
        m_dev_ = std::move(dev);
        m_cmdpool_ = std::move(cmdpool);
    }

    virtual void setAttribute(
        const std::unordered_map<std::string, std::string> &attributes) {
        if (!attributes.empty()) {
            for (const auto &attr : attributes) {
                std::cout << "Warning: Unused attribute " << attr.first
                          << " in operator." << std::endl;
            }
        }
    }
    virtual inline std::vector<int> parse_attr_list(const std::string &str) {
        std::vector<int> result;
        if (str.front() == '[' && str.back() == ']') {
            std::string content = str.substr(1, str.size() - 2);
            std::stringstream ss(content);
            std::string item;
            while (std::getline(ss, item, ',')) {
                result.push_back(std::stoi(item));
            }
        }
        return result;
    }

    virtual OpType get_type() { return type_; }

    virtual void
    execute(std::vector<std::shared_ptr<core::ITensor>> inputs,
            std::vector<std::shared_ptr<core::ITensor>> outputs) = 0;

    virtual void apply(std::vector<std::shared_ptr<core::ITensor>> inputs,
                       std::vector<std::shared_ptr<core::ITensor>> outputs) = 0;

  protected:
    VkPhysicalDevice m_phydev_;
    std::shared_ptr<VulkanDevice> m_dev_;
    std::shared_ptr<VulkanCommandPool> m_cmdpool_;
    OpType type_;

    std::vector<std::shared_ptr<VulkanImage>> inputImages_;
    std::shared_ptr<VulkanImage> outputImage_;

    virtual void submit(const unsigned char *spv, unsigned int spv_len,
                        int out_width, int out_height) = 0;
};

} // namespace ops
} // namespace vkop
#endif // OPS_OPERATOR_HPP_
