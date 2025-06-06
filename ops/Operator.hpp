// Copyright 2025 @junka
#ifndef OPS_OPERATOR_HPP_
#define OPS_OPERATOR_HPP_

#include <vector>

#include "Tensor.hpp"
#include "VulkanCommandPool.hpp"
#include "VulkanDevice.hpp"

#define UP_DIV(x, y) (((x) + (y)-1) / (y))

namespace vkop {

namespace ops {

class Operator {
  public:
    Operator() = default;
    virtual ~Operator() = default;
    Operator(const Operator &) = delete;
    Operator &operator=(const Operator &) = delete;
    Operator(Operator &&) = delete;
    Operator &operator=(Operator &&) = delete;

    virtual void set_runtime_device(VkPhysicalDevice phydev,
                                    std::shared_ptr<VulkanDevice> &dev,
                                    VulkanCommandPool *cmdpool) {
        m_phydev_ = phydev;
        m_dev_ = dev.get();
        m_cmdpool_ = cmdpool;
    }

    virtual void
    execute(std::vector<std::shared_ptr<core::Tensor<float>>> inputs,
            std::vector<std::shared_ptr<core::Tensor<float>>> outputs) = 0;
    virtual void
    execute(std::vector<std::shared_ptr<core::Tensor<int>>> inputs,
            std::vector<std::shared_ptr<core::Tensor<int>>> outputs) = 0;

  protected:
    VkPhysicalDevice m_phydev_;
    VulkanDevice *m_dev_;
    VulkanCommandPool *m_cmdpool_;

    virtual void submit(const unsigned char *spv, unsigned int spv_len,
                        int out_width, int out_height) = 0;
};

} // namespace ops

} // namespace vkop
#endif // OPS_OPERATOR_HPP_
