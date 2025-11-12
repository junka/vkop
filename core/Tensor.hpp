// Copyright 2025 @junka
#ifndef CORE_TENSOR_HPP_
#define CORE_TENSOR_HPP_

#include <bits/c++config.h>
#include <cstdint>
#include <iostream>
#include <memory>
#include <vector>

#include "vulkan/VulkanCommandBuffer.hpp"
#include "vulkan/VulkanCommandPool.hpp"
#include "vulkan/VulkanDevice.hpp"
#include "vulkan/VulkanImage.hpp"
#include "vulkan/VulkanResource.hpp"

#define UP_DIV(x, y) (((x) + (y)-1) / (y))

namespace vkop {
namespace core {

template <typename T> class Tensor;

class ITensor {
  public:
    virtual ~ITensor() = default;
    virtual const std::type_info &dtype() const = 0;

    template <typename T> Tensor<T> *as() const {
        if (dtype() == typeid(T)) {
            return static_cast<Tensor<T> *>(this);
        }
        return nullptr;
    }

    int num_dims() { return dims_.size(); };
    bool is_on_GPU() const { return converted_; }
    void toGPU() { converted_ = true; }
    void toCPU() { converted_ = false; }

  protected:
    std::vector<int> dims_;
    bool converted_ = false;
};

template <typename T> class Tensor : public ITensor {
  public:
    // empty
    Tensor() = default;

    // nchw
    Tensor(int n, int c, int h, int w) {
        dims_ = std::vector<int>{n, c, h, w};
        ele_size_ = sizeof(T);
        fp16_ = (sizeof(T) == 2);
        size_ = ele_size_ * n * c * h * w;
        data_.resize(n * c * h * w);
    }

    // nchw in vector
    explicit Tensor(const std::vector<int> &dims) {
        dims_ = dims;
        ele_size_ = sizeof(T);
        fp16_ = (sizeof(T) == 2);
        size_ = ele_size_;
        for (auto d : dims_) {
            size_ *= d;
        }
        data_.resize(size_ / ele_size_);
    }

    // nchw in vector
    explicit Tensor(const std::vector<uint32_t> &dims) {
        dims_ = std::vector<int>(dims.begin(), dims.end());
        ele_size_ = sizeof(T);
        fp16_ = (sizeof(T) == 2);
        size_ = ele_size_;
        for (auto d : dims_) {
            size_ *= d;
        }
        data_.resize(size_ / ele_size_);
    }

    const std::type_info &dtype() const override { return typeid(T); }

    void resize(int n, int c, int h, int w) {
        dims_ = std::vector<int>{n, c, h, w};
        ele_size_ = sizeof(T);
        fp16_ = (sizeof(T) == 2);
        size_ = ele_size_ * n * c * h * w;
        data_.resize(n * c * h * w);
    }

    void resize(const std::vector<int> &dims) {
        dims_ = dims;
        ele_size_ = sizeof(T);
        fp16_ = (sizeof(T) == 2);
        size_ = ele_size_;
        for (auto d : dims_) {
            size_ *= d;
        }
        data_.resize(size_ / ele_size_);
    }

    void printTensorShape() const {
        std::cout << "Tensor dim " << dims_.size() << ", shape: [";
        for (auto d : dims_) {
            std::cout << d << " ";
        }
        std::cout << "], size " << size_ << std::endl;
    }

    /**
     * @brief Deleted copy constructor to prevent copying of tensor objects.
     */
    Tensor(const Tensor &tensor) = delete;
    /**
     * @brief Deleted move constructor to prevent moving of tensor objects.
     */
    Tensor(const Tensor &&tensor) = delete;
    /**
     * @brief Deleted copy assignment operator to prevent copying of tensor
     * objects.
     */
    Tensor &operator=(const Tensor &) = delete;

    /**
     * @brief Deleted move assignment operator to prevent moving of tensor
     * objects.
     */
    Tensor &operator=(const Tensor &&) = delete;

    T &operator[](std::size_t index) { return data_[index]; }

    const T &operator[](std::size_t index) const { return data_[index]; }

    T &at(std::size_t index) {
        if (index >= data_.size()) {
            throw std::out_of_range("Index out of range");
        }
        return data_[index];
    }

    const T &at(std::size_t index) const {
        if (index >= data_.size()) {
            throw std::out_of_range("Index out of range");
        }
        return data_[index];
    }

    std::vector<int> getTensorShape() { return dims_; }

    T *data() { return data_.data(); }

    int size() { return size_; }
    int num_elements() { return size_ / ele_size_; }

    // void *map() { return nullptr; }
    // void unmap() { (void)vkobj_; }

    /**
     * @brief Converts a tensor into an RGBA format suitable for Vulkan image
     * processing.
     *
     * This function assumes the tensor data is stored in a specific layout and
     * converts it into an RGBA format, where each pixel contains four channels
     * (R, G, B, A). The conversion is necessary because Vulkan images typically
     * use RGBA formats for efficient rendering and processing. The function
     * handles cases where the depth of the tensor is less than 4 by padding the
     * remaining channels with zeros.
     *
     * @tparam T The data type of the tensor elements.
     * @return A vector containing the converted RGBA data.
     *
     * @details
     * - The function first ensures that the resource type is a Vulkan image.
     * - It dynamically casts the Vulkan resource to a VulkanImage object.
     * - The tensor dimensions (batch, depth, height, width) are used to
     * calculate strides and the real dimensions required for RGBA conversion.
     * - The tensor data is rearranged into RGBA format, with padding for
     * channels beyond the tensor's depth.
     * - The resulting RGBA data is stored in a temporary vector and returned.
     *
     * @throws std::runtime_error If the Vulkan resource cannot be cast to a
     * VulkanImage.
     *
     * @note This function is designed for tensors used in Vulkan-based GPU
     * processing pipelines. The conversion ensures compatibility with Vulkan's
     * image formats and facilitates efficient GPU operations.
     */
    std::vector<T> *convertTensorToRGBA() {
        auto img = vkobj_.lock();
        // assert(obj->getResourceType() == ResourceType::VK_IMAGE);
        // auto img = std::dynamic_pointer_cast<VulkanImage>(obj);
        if (!img) {
            throw std::runtime_error(
                "Failed to cast VulkanResource to VulkanImage");
        }
        auto batch = dims_[0];
        auto depth = dims_[1];
        auto height = dims_[2];
        auto width = dims_[3];

        int stride_w = 1;
        int stride_h = width;
        int stride_c = width * height;
        int stride_n = width * height * depth;
        int realdepth = UP_DIV(depth, 4);
        int realwidth = width * realdepth;
        int realheight = height * batch;

        auto ptr = new std::vector<T>(realheight * realwidth *
                                      img->getImageChannelNum() *
                                      img->getImageChannelSize());
        uint32_t row_pitch =
            realwidth * img->getImageChannelNum() * img->getImageChannelSize();
        T *dst = ptr->data();
        for (int b = 0; b < batch; b++) {
            T *batchstart =
                reinterpret_cast<T *>(reinterpret_cast<uint8_t *>(ptr->data()) +
                                      b * height * row_pitch);
            for (int c = 0; c < realdepth; c++) {
                dst = batchstart + c * 4 * width;
                for (int h = 0; h < height; h++) {
                    for (int w = 0; w < width; w++) {
                        int offset = b * stride_n + 4 * c * stride_c +
                                     h * stride_h + w * stride_w;

                        dst[w * 4 + 0] = data_[offset];
                        dst[w * 4 + 1] = (4 * c + 1 < depth)
                                             ? data_[stride_c + offset]
                                             : 0.0F;
                        dst[w * 4 + 2] = (4 * c + 2 < depth)
                                             ? data_[2 * stride_c + offset]
                                             : 0.0F;
                        dst[w * 4 + 3] = (4 * c + 3 < depth)
                                             ? data_[3 * stride_c + offset]
                                             : 0.0F;
                    }
                    dst = reinterpret_cast<T *>(
                        reinterpret_cast<uint8_t *>(dst) + row_pitch);
                }
            }
        }
        return ptr;
    }

    void copyToGPU(const std::shared_ptr<VulkanDevice> &dev,
                   const std::shared_ptr<VulkanCommandPool> &cmdpool) {
        auto img = vkobj_.lock();
        VkDevice device = dev->getLogicalDevice();
#ifdef VK_EXT_host_image_copy
        if (m_dev_->is_support_host_image_copy()) {
            if (dims_.size() < 3 || is_on_GPU()) {
                return;
            }
            auto ptr = convertTensorToRGBA();
            img->hostImageCopyToDevice(ptr->data());
            toGPU();
            delete ptr;
        } else
#endif
        {
            VulkanCommandBuffer cmdstg(device, cmdpool->getCommandPool());
            cmdstg.begin();
            if (dims_.size() < 3 || is_on_GPU()) {
                return;
            }
            auto ptr = convertTensorToRGBA();
            img->stagingBufferCopyToImage(cmdstg.get(), ptr->data());
            toGPU();
            delete ptr;
            cmdstg.end();
            cmdstg.submit(dev->getComputeQueue());
        }
        VulkanCommandBuffer cmd(device, cmdpool->getCommandPool());
        cmd.begin();
        img->readBarrier(cmd.get());
        cmd.end();
        cmd.submit(dev->getComputeQueue());
    }

    void convertRGBAToTensor(T *ptr) {
        auto batch = dims_[0];
        auto depth = dims_[1];
        auto height = dims_[2];
        auto width = dims_[3];

        int stride_w = 1;
        int stride_h = width;
        int stride_c = width * height;
        int stride_n = width * height * depth;

        int realdepth = UP_DIV(depth, 4);
        int realwidth = width * UP_DIV(depth, 4);

        uint32_t row_pitch = realwidth * 4 * sizeof(T);
        T *dst;
        for (int b = 0; b < batch; b++) {
            T *batchstart = reinterpret_cast<T *>(
                reinterpret_cast<uint8_t *>(ptr) + b * height * row_pitch);
            for (int c = 0; c < realdepth; c++) {
                dst = batchstart + c * width * 4;
                for (int h = 0; h < height; h++) {
                    for (int w = 0; w < width; w++) {
                        int offset = b * stride_n + 4 * c * stride_c +
                                     h * stride_h + w * stride_w;
                        data_[offset] = dst[w * 4 + 0];
                        if (4 * c + 1 < depth) {
                            data_[stride_c + offset] = dst[w * 4 + 1];
                        }
                        if (4 * c + 2 < depth) {
                            data_[stride_c * 2 + offset] = dst[w * 4 + 2];
                        }
                        if (4 * c + 3 < depth) {
                            data_[stride_c * 3 + offset] = dst[w * 4 + 3];
                        }
                    }
                    dst = reinterpret_cast<T *>(
                        reinterpret_cast<uint8_t *>(dst) + row_pitch);
                }
            }
        }
    }

    void copyToCPU(const std::shared_ptr<VulkanDevice> &dev,
                   const std::shared_ptr<VulkanCommandPool> &cmdpool) {
        auto img = vkobj_.lock();
        VkDevice device = dev->getLogicalDevice();

        int batch = dims_[0];
        int depth = dims_[1];
        int out_height = dims_[2];
        int out_width = dims_[3];
        int realwidth = out_width * UP_DIV(depth, 4);
        int realheight = out_height * batch;

        auto ptr = new std::vector<T>((realheight * realwidth * 4));

#ifdef VK_EXT_host_image_copy
        if (m_dev->is_support_host_image_copy()) {
            img->hostImageCopyToHost(ptr->data());
        } else
#endif
        {
            VulkanCommandBuffer cmd(device, cmdpool->getCommandPool());
            cmd.begin();
            VulkanCommandBuffer cmdstg1(device, cmdpool->getCommandPool());
            cmdstg1.begin();
            img->stagingBufferCopyToHost(cmdstg1.get());
            cmdstg1.end();
            cmdstg1.submit(dev->getComputeQueue());
            img->readStaingBuffer(ptr->data());
        }

        convertRGBAToTensor(ptr->data());
        toCPU();
        delete ptr;
    }

    std::shared_ptr<VulkanImage> make_vkimg(std::shared_ptr<VulkanDevice> &vd,
                                            uint32_t flags) {
        auto vkimg = std::make_shared<VulkanImage>(
            vd,
            VkExtent3D{static_cast<uint32_t>(dims_[3] * UP_DIV(dims_[1], 4)),
                       static_cast<uint32_t>(dims_[2] * dims_[0]), 1},
            flags, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            (fp16_ ? VK_FORMAT_R16G16B16A16_SFLOAT
                   : VK_FORMAT_R32G32B32A32_SFLOAT));

        vkobj_ = vkimg;
        return vkimg;
    }

  private:
    std::vector<T> data_;

    int ele_size_;
    int size_;

    bool fp16_;

    // std::weak_ptr<VulkanResource> vkobj_;
    std::weak_ptr<VulkanImage> vkobj_;
};

template <typename T>
inline std::shared_ptr<Tensor<T>>
as_tensor(const std::shared_ptr<ITensor> &ptr) {
    return std::dynamic_pointer_cast<Tensor<T>>(ptr);
}

} // namespace core
} // namespace vkop

#endif // CORE_TENSOR_HPP_
