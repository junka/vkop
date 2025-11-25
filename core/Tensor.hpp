// Copyright 2025 @junka
#ifndef CORE_TENSOR_HPP_
#define CORE_TENSOR_HPP_

#include "vulkan/VulkanBuffer.hpp"
#include "vulkan/VulkanCommandBuffer.hpp"
#include "vulkan/VulkanCommandPool.hpp"
#include "vulkan/VulkanDevice.hpp"
#include "vulkan/VulkanImage.hpp"
#include "vulkan/VulkanResource.hpp"

#include <chrono>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <thread>
#include <vector>

#define UP_DIV(x, y) (((x) + (y)-1) / (y))

namespace vkop {
namespace core {

template <typename T> class Tensor;

class ITensor {
  public:
    virtual ~ITensor() = default;
    virtual const std::type_info &dtype() const = 0;

    int num_dims() { return dims_.size(); };
    bool is_on_GPU() const { return converted_; }
    void toGPU() { converted_ = true; }
    void toCPU() { converted_ = false; }
    void ref_inc() { ref_cnt_++; }
    int ref_cnt() const { return ref_cnt_; }
    void set_ref_cnt_forever() { ref_cnt_ = std::numeric_limits<int>::max(); }
    std::vector<int> getShape() { return dims_; }

    static float fp16_to_fp32(uint16_t h) {
        uint32_t sign = (static_cast<uint32_t>(h) & 0x8000) << 16;
        uint32_t exponent = (h & 0x7C00) >> 10;
        uint32_t mantissa = h & 0x03FF;

        if (exponent == 0) {
            if (mantissa == 0) {
                // ±0
                uint32_t r = sign;
                float f;
                std::memcpy(&f, &r, sizeof(f));
                return f;
            }
            // Subnormal
            while ((mantissa & 0x0400) == 0) {
                mantissa <<= 1;
                exponent--;
            }
            exponent++;
            mantissa &= 0x03FF;

        } else if (exponent == 31) {
            // Inf or NaN
            uint32_t r = sign | 0x7F800000 | (mantissa << 13);
            float f;
            std::memcpy(&f, &r, sizeof(f));
            return f;
        }

        uint32_t r = sign | ((exponent + 127 - 15) << 23) | (mantissa << 13);
        float f;
        std::memcpy(&f, &r, sizeof(f));
        return f;
    }
    static uint16_t fp32_to_fp16(float f) {
        uint32_t x;
        std::memcpy(&x, &f, sizeof(f));

        uint16_t sign = (x >> 16) & 0x8000;
        int16_t exponent = ((x >> 23) & 0xFF) - 127 + 15; // adjust bias
        uint32_t mantissa = x & 0x007FFFFF;

        if (exponent <= 0) {
            // Zero or subnormal
            if (exponent < -10)
                return sign;
            mantissa = (mantissa | 0x00800000) >> (1 - exponent);
            return sign | (mantissa + 0x00001000) >> 13;
        }
        if (exponent >= 31) {
            // Infinity or NaN
            return sign | 0x7C00;
        }
        // Normal number
        return sign | (exponent << 10) | (mantissa >> 13);
    }

  protected:
    int size_;
    int ref_cnt_ = 0;
    bool converted_ = false;
    std::vector<int> dims_;
};

template <typename T> class Tensor : public ITensor {
  public:
    // empty
    Tensor() = default;

    // nchw
    Tensor(int n, int c, int h, int w) {
        dims_ = std::vector<int>{n, c, h, w};
        size_ = sizeof(T) * n * c * h * w;
    }

    // nchw in vector
    explicit Tensor(const std::vector<int> &dims) {
        dims_ = dims;
        size_ = dims_.size() ? sizeof(T) : 0;
        for (auto d : dims_) {
            size_ *= d;
        }
    }

    // nchw in vector
    explicit Tensor(const std::vector<uint32_t> &dims) {
        dims_ = std::vector<int>(dims.begin(), dims.end());
        size_ = dims_.size() ? sizeof(T) : 0;
        for (auto d : dims_) {
            size_ *= d;
        }
    }

    const std::type_info &dtype() const override { return typeid(T); }

    void resize(int n, int c, int h, int w) {
        dims_ = std::vector<int>{n, c, h, w};
        size_ = sizeof(T) * n * c * h * w;
        if (!is_on_GPU())
            data_.resize(n * c * h * w);
    }

    void resize(const std::vector<int> &dims) {
        dims_ = dims;
        size_ = dims_.size() ? sizeof(T) : 0;
        for (auto d : dims_) {
            size_ *= d;
        }
        if (!is_on_GPU())
            data_.resize(size_ / sizeof(T));
    }

    void resize(int len) {
        if (len == 0) {
            if (is_on_GPU()) {
                vkobj_.reset();
                vkobj_ = nullptr;
            } else {
                data_.clear();
                data_.shrink_to_fit();
            }
            size_ = 0;
            dims_.clear();
        }
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

    int size() { return size_; }
    int num_elements() { return size_ / sizeof(T); }

    std::shared_ptr<VulkanBuffer>
    as_storage_buffer(std::shared_ptr<VulkanDevice> &vd) {
        if (!vkobj_) {
            vkobj_ = std::make_shared<VulkanBuffer>(
                vd, size_,
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                    VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                    VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        }
        return std::dynamic_pointer_cast<VulkanBuffer>(vkobj_);
    }
    std::shared_ptr<VulkanBuffer>
    as_uniform_buffer(std::shared_ptr<VulkanDevice> &vd) {
        if (!vkobj_) {
            vkobj_ = std::make_shared<VulkanBuffer>(
                vd, size_,
                VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT |
                    VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        }
        return std::dynamic_pointer_cast<VulkanBuffer>(vkobj_);
    }
    std::shared_ptr<VulkanImage>
    as_input_image(std::shared_ptr<VulkanDevice> &vd,
                   const std::shared_ptr<VulkanCommandBuffer> &cmd) {
        if (!vkobj_) {
            int exflags = 0;
            if (vd->is_support_host_image_copy()) {
#ifdef VK_EXT_host_image_copy
                exflags |= VK_IMAGE_USAGE_HOST_TRANSFER_BIT;
#endif
            }
            make_vkimg(vd, VK_IMAGE_USAGE_SAMPLED_BIT |
                               VK_IMAGE_USAGE_STORAGE_BIT |
                               VK_IMAGE_USAGE_TRANSFER_DST_BIT | exflags);
        }
        auto img = std::dynamic_pointer_cast<VulkanImage>(vkobj_);
#ifdef VK_EXT_host_image_copy
        if (vd->is_support_host_image_copy()) {
            img->hostImaggeTransition(VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        } else
#endif
        {
            if (cmd) {
                // cmd->begin();
                img->readBarrier(cmd->get());
                // cmd->end();
            }
        }

        return img;
    }

    std::shared_ptr<VulkanImage>
    as_output_image(std::shared_ptr<VulkanDevice> &vd,
                    const std::shared_ptr<VulkanCommandBuffer> &cmd) {
        if (!vkobj_) {
            int exflags = 0;
            if (vd->is_support_host_image_copy()) {
#ifdef VK_EXT_host_image_copy
                exflags |= VK_IMAGE_USAGE_HOST_TRANSFER_BIT;
#endif
            }
            make_vkimg(vd, VK_IMAGE_USAGE_STORAGE_BIT |
                               VK_IMAGE_USAGE_SAMPLED_BIT |
                               VK_IMAGE_USAGE_TRANSFER_SRC_BIT | exflags);
        }
        auto img = std::dynamic_pointer_cast<VulkanImage>(vkobj_);
#ifdef VK_EXT_host_image_copy
        if (vd->is_support_host_image_copy()) {
            if (vd->checkHostImageCopyDstLayoutSupport(
                    VK_IMAGE_LAYOUT_GENERAL)) {
                img->hostImaggeTransition(VK_IMAGE_LAYOUT_GENERAL);
            } else {
                img->hostImaggeTransition(VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
            }
        } else
#endif
        {
            if (cmd) {
                // cmd->begin();
                img->writeBarrier(cmd->get());
                // cmd->end();
            }
        }

        return img;
    }
    void copyToGPU(const std::shared_ptr<VulkanDevice> &dev,
                   const std::shared_ptr<VulkanCommandPool> &cmdpool,
                   T *data = nullptr) {
        if (is_on_GPU()) {
            return;
        }
        if (vkobj_->getResourceType() == ResourceType::VK_IMAGE) {
            copyToGPUImage(dev, cmdpool, data);
        } else {
            copyToGPUBuffer(dev, cmdpool, data);
        }
        if (!data) {
            data_.clear();
            data_.shrink_to_fit();
        }
    }

    void copyToCPU(const std::shared_ptr<VulkanDevice> &dev,
                   const std::shared_ptr<VulkanCommandPool> &cmdpool) {
        if (!is_on_GPU()) {
            printf("not on GPU\n");
            return;
        }
        data_.resize(num_elements());
        if (vkobj_->getResourceType() == ResourceType::VK_IMAGE) {
            copyImageToCPU(dev, cmdpool);
        } else {
            copyBufferToCPU(dev, cmdpool);
        }
    }

    void fillToCPU(std::vector<T> &data) {
        data_.resize(num_elements());
        memcpy(data_.data(), data.data(), size_);
        toCPU();
    }
    void fillToCPU(T *data) {
        data_.resize(num_elements());
        memcpy(data_.data(), data, size_);
        toCPU();
    }

    // implicity fp convertor
    void fillFP32ToCPU(std::vector<float> &data) {
        if (typeid(T) == typeid(uint16_t)) {
            data_.resize(num_elements());
            for (int i = 0; i < num_elements(); i++) {
                data_[i] = fp32_to_fp16(data[i]);
            }
        } else if (typeid(T) == typeid(float)) {
            data_.resize(num_elements());
            memcpy(data_.data(), data.data(), size_);
            toCPU();
        } else {
            throw std::runtime_error("not convertedto fp16");
        }
    }
    void fillFP32ToCPU(float *data) {
        if (typeid(T) == typeid(uint16_t)) {
            data_.resize(num_elements());
            for (int i = 0; i < num_elements(); i++) {
                data_[i] = fp32_to_fp16(data[i]);
            }
        } else if (typeid(T) == typeid(float)) {
            data_.resize(num_elements());
            memcpy(data_.data(), data, size_);
            toCPU();
        } else {
            throw std::runtime_error("not convertedto fp16");
        }
    }

    void reserveOnCPU() {
        if (is_on_GPU()) {
            printf("Should not call reserveOnCPU when Tensor is on GPU");
            return;
        }
        data_.resize(num_elements());
        toCPU();
    }

  private:
    std::shared_ptr<VulkanResource> vkobj_;
    std::vector<T> data_;

    void make_vkimg(std::shared_ptr<VulkanDevice> &vd, uint32_t flags) {
        if (vkobj_) {
            return;
        }
        VkFormat format =
            ((sizeof(T) == 1)
                 ? VK_FORMAT_R8G8B8A8_UINT
                 : ((sizeof(T) == 2) ? VK_FORMAT_R16G16B16A16_SFLOAT
                                     : VK_FORMAT_R32G32B32A32_SFLOAT));
        vkobj_ = std::make_shared<VulkanImage>(
            vd,
            VkExtent3D{static_cast<uint32_t>(dims_[3] * UP_DIV(dims_[1], 4)),
                       static_cast<uint32_t>(dims_[2] * dims_[0]), 1},
            flags, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, format);
    }

    void copyToGPUBuffer(const std::shared_ptr<VulkanDevice> &dev,
                         const std::shared_ptr<VulkanCommandPool> &cmdpool,
                         T *src = nullptr) {
        auto buffer = std::dynamic_pointer_cast<VulkanBuffer>(vkobj_);
        VkDevice device = dev->getLogicalDevice();

        VulkanCommandBuffer cmd(device, cmdpool->getCommandPool(),
                                cmdpool->getSemaphore());
        auto stpool = cmdpool->getStagingBufferPool();
        uint64_t completed_timeline_value =
            cmdpool->getCompletedTimelineValue();
        stpool->reclaimCompleted(completed_timeline_value);
        auto *buff = stpool->getBuffer();
        auto b = stpool->allocate(size_);
        if (!b) {
            printf("stpool alloc failed\n");
            return;
        }
        if (src)
            memcpy(b->ptr, src, size_);
        else {
            memcpy(b->ptr, data_.data(), size_);
        }

        cmd.begin();
        buffer->copyStageBufferToBuffer(cmd.get(), buff, b->offset);
        buffer->readBarrier(cmd.get());
        cmd.end();
        uint64_t submit_value = cmdpool->getNextSubmitValue();
        cmd.submit(dev->getComputeQueue(), submit_value);
        stpool->markSubmit(submit_value);
        cmd.wait(submit_value);
        toGPU();
    }
    void copyToGPUImage(const std::shared_ptr<VulkanDevice> &dev,
                        const std::shared_ptr<VulkanCommandPool> &cmdpool,
                        T *src = nullptr) {
        auto img = std::dynamic_pointer_cast<VulkanImage>(vkobj_);
        VkDevice device = dev->getLogicalDevice();

        if (dims_.size() < 3) {
            return;
        }
        VulkanCommandBuffer cmd(device, cmdpool->getCommandPool(),
                                cmdpool->getSemaphore());

#ifdef VK_EXT_host_image_copy
        if (dev->is_support_host_image_copy()) {
            if (dims_.size() < 3 || is_on_GPU()) {
                return;
            }
            std::vector<T> ptr(img->getImageSize());
            convertTensorToRGBA(ptr.data(), src);
            img->hostImageCopyToDevice(ptr.data());
            cmd.begin();
            img->readBarrier(cmd.get());
            cmd.end();
            cmd.submit(dev->getComputeQueue(), cmdpool->getFence());
        } else
#endif
        {
            auto imagesize = img->getImageSize();
            auto stpool = cmdpool->getStagingBufferPool();
            uint64_t completed_timeline_value =
                cmdpool->getCompletedTimelineValue();
            stpool->reclaimCompleted(completed_timeline_value);
            auto *buff = stpool->getBuffer();
            auto b = stpool->allocate(imagesize);
            if (!b) {
                printf("stpool alloc failed\n");
                return;
            }
            convertTensorToRGBA(static_cast<T *>(b->ptr), src);

            cmd.begin();
            img->copyBufferToImage(cmd.get(), buff, b->offset);
            img->readBarrier(cmd.get());
            cmd.end();
            uint64_t submit_value = cmdpool->getNextSubmitValue();
            cmd.submit(dev->getComputeQueue(), submit_value);
            stpool->markSubmit(submit_value);
            cmd.wait(submit_value);
        }
        toGPU();
    }
    void copyBufferToCPU(const std::shared_ptr<VulkanDevice> &dev,
                         const std::shared_ptr<VulkanCommandPool> &cmdpool) {
        auto buffer = std::dynamic_pointer_cast<VulkanBuffer>(vkobj_);

        VkDevice device = dev->getLogicalDevice();
        VulkanCommandBuffer cmd(device, cmdpool->getCommandPool(),
                                cmdpool->getSemaphore());

        auto stpool = cmdpool->getStagingBufferPool();
        uint64_t completed_timeline_value =
            cmdpool->getCompletedTimelineValue();
        stpool->reclaimCompleted(completed_timeline_value);
        auto *buff = stpool->getBuffer();
        auto b = stpool->allocate(size_);
        if (!b) {
            printf("stpool alloc failed\n");
            return;
        }

        cmd.begin();
        buffer->copyBufferToStageBuffer(cmd.get(), buff, b->offset);
        cmd.end();
        uint64_t submit_value = cmdpool->getNextSubmitValue();
        cmd.submit(dev->getComputeQueue(), submit_value);
        stpool->markSubmit(submit_value);
        cmd.wait(submit_value);
        std::memcpy(data_.data(), b->ptr, size_);
        toCPU();
    }
    void copyImageToCPU(const std::shared_ptr<VulkanDevice> &dev,
                        const std::shared_ptr<VulkanCommandPool> &cmdpool) {
        auto img = std::dynamic_pointer_cast<VulkanImage>(vkobj_);

        VkDevice device = dev->getLogicalDevice();

        int batch = dims_[0];
        int depth = dims_[1];
        int out_height = dims_[2];
        int out_width = dims_[3];
        int realwidth = out_width * UP_DIV(depth, 4);
        int realheight = out_height * batch;

        auto imagesize = img->getImageSize();
        assert(realheight * realwidth * 4 * sizeof(T) ==
               static_cast<size_t>(imagesize));

#ifdef VK_EXT_host_image_copy
        if (dev->is_support_host_image_copy()) {
            std::vector<T> rgba((realheight * realwidth * 4));
            img->hostImageCopyToHost(rgba.data());
            convertRGBAToTensor(rgba.data());
        } else
#endif
        {
            VulkanCommandBuffer cmd(device, cmdpool->getCommandPool(),
                                    cmdpool->getSemaphore());

            auto stpool = cmdpool->getStagingBufferPool();
            uint64_t completed_timeline_value =
                cmdpool->getCompletedTimelineValue();
            stpool->reclaimCompleted(completed_timeline_value);
            auto *buff = stpool->getBuffer();
            auto b = stpool->allocate(imagesize);
            if (!b) {
                printf("stpool alloc failed\n");
                return;
            }

            cmd.begin();
            img->copyImageToBuffer(cmd.get(), buff, b->offset);
            cmd.end();
            uint64_t submit_value = cmdpool->getNextSubmitValue();
            cmd.submit(dev->getComputeQueue(), submit_value);
            stpool->markSubmit(submit_value);
            cmd.wait(submit_value);
            convertRGBAToTensor(static_cast<T *>(b->ptr));
        }

        toCPU();
    }

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
     * @tparam T The data type of the tensor elements. Size of it make be equal
     *  to getImageSize()
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
    void convertTensorToRGBA(T *ptr, T *src = nullptr) {
        auto img = std::dynamic_pointer_cast<VulkanImage>(vkobj_);
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

        uint32_t row_pitch =
            realwidth * img->getImageChannelNum() * img->getImageChannelSize();
        T *dst = ptr;
        for (int b = 0; b < batch; b++) {
            T *batchstart = reinterpret_cast<T *>(
                reinterpret_cast<uint8_t *>(ptr) + b * height * row_pitch);
            for (int c = 0; c < realdepth; c++) {
                dst = batchstart + c * 4 * width;
                for (int h = 0; h < height; h++) {
                    for (int w = 0; w < width; w++) {
                        int offset = b * stride_n + 4 * c * stride_c +
                                     h * stride_h + w * stride_w;

                        dst[w * 4 + 0] = src ? src[offset] : data_[offset];
                        dst[w * 4 + 1] = (4 * c + 1 < depth)
                                             ? (src ? src[stride_c + offset]
                                                    : data_[stride_c + offset])
                                             : 0.0F;
                        dst[w * 4 + 2] =
                            (4 * c + 2 < depth)
                                ? (src ? src[2 * stride_c + offset]
                                       : data_[2 * stride_c + offset])
                                : 0.0F;
                        dst[w * 4 + 3] =
                            (4 * c + 3 < depth)
                                ? (src ? src[3 * stride_c + offset]
                                       : data_[3 * stride_c + offset])
                                : 0.0F;
                    }
                    dst = reinterpret_cast<T *>(
                        reinterpret_cast<uint8_t *>(dst) + row_pitch);
                }
            }
        }
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
};

template <typename T>
inline std::shared_ptr<Tensor<T>>
as_tensor(const std::shared_ptr<ITensor> &ptr) {
    return std::dynamic_pointer_cast<Tensor<T>>(ptr);
}

} // namespace core
} // namespace vkop

#endif // CORE_TENSOR_HPP_
