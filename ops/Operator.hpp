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

    template <typename T>
    void copyTensorToImages(
        const std::vector<std::shared_ptr<core::ITensor>> &inputs) {
        VkDevice device = m_dev_->getLogicalDevice();
        int cnt = 0;
#ifdef VK_EXT_host_image_copy
        if (m_dev_->is_support_host_image_copy()) {
            for (const auto &input : inputs) {
                if (input == nullptr) {
                    continue;
                }

                auto t = core::as_tensor<T>(input);
                if (t->num_dims() < 3) {
                    continue;
                }
                inputImages_[cnt++]->hostImageCopyToDevice(
                    t->convertTensorToRGBA().data());
            }
        } else
#endif
        {
            VulkanCommandBuffer cmdstg(device, m_cmdpool_->getCommandPool());
            cmdstg.begin();
            for (const auto &input : inputs) {
                if (input == nullptr) {
                    continue;
                }
                auto t = core::as_tensor<T>(input);
                if (t->num_dims() < 3) {
                    continue;
                }
                inputImages_[cnt++]->stagingBufferCopyToImage(
                    cmdstg.get(), t->convertTensorToRGBA().data());
            }
            cmdstg.end();
            cmdstg.submit(m_dev_->getComputeQueue());
        }
        VulkanCommandBuffer cmd(device, m_cmdpool_->getCommandPool());
        cmd.begin();
        for (int i = 0; i < cnt; i++) {
            inputImages_[i]->readBarrier(cmd.get());
        }
        cmd.end();
        cmd.submit(m_dev_->getComputeQueue());
    }
    template <typename T>
    void copyImageToTensor(const std::shared_ptr<core::Tensor<T>> &output) {
        VkDevice device = m_dev_->getLogicalDevice();

        auto dim = output->getTensorShape();
        int batch = dim[0];
        int depth = dim[1];
        int out_height = dim[2];
        int out_width = dim[3];
        int realwidth = out_width * UP_DIV(depth, 4);
        int realheight = out_height * batch;

        std::vector<T> tmp(realheight * realwidth * 4);
        T *ptr = tmp.data();
#ifdef VK_EXT_host_image_copy
        if (m_dev->is_support_host_image_copy()) {
            outputImage->hostImageCopyToHost(ptr);
        } else
#endif
        {
            VulkanCommandBuffer cmd(device, m_cmdpool_->getCommandPool());
            cmd.begin();
            VulkanCommandBuffer cmdstg1(device, m_cmdpool_->getCommandPool());
            cmdstg1.begin();
            outputImage_->stagingBufferCopyToHost(cmdstg1.get());
            cmdstg1.end();
            cmdstg1.submit(m_dev_->getComputeQueue());
            outputImage_->readStaingBuffer(ptr);
        }

        output->convertRGBAToTensor(ptr);
    }

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
