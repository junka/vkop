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
    virtual const std::type_info &dtype() const = 0;

    int num_dims() {
        int count = 4;
        while (count > 0 && dims_[count - 1] == 0) {
            count--;
        }
        return count;
    }

    int size() const { return size_; }
    bool is_on_GPU() const { return converted_; }
    void toGPU() { converted_ = true; }
    void toCPU() { converted_ = false; }
    void ref_inc() { ref_cnt_++; }
    int ref_cnt() const { return ref_cnt_; }
    void set_ref_cnt_forever() {
        ref_cnt_ = std::numeric_limits<uint16_t>::max();
    }
    void set_transpose() { transpose_ = true; }
    bool get_transpose() const { return transpose_; }
    bool get_pack() const { return pack_; }
    std::vector<int> getShape() {
        if (dims_[3]) {
            return std::vector<int>{dims_[0], dims_[1], dims_[2], dims_[3]};
        }
        if (dims_[2]) {
            return std::vector<int>{dims_[0], dims_[1], dims_[2]};
        }
        if (dims_[1]) {
            return std::vector<int>{dims_[0], dims_[1]};
        }
        return std::vector<int>{dims_[0]};
    }

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
    int dims_[4];
    int size_ = 0;
    uint16_t ref_cnt_ = 0;
    bool transpose_ = false;
    bool pack_ = false;
    bool converted_ = false;
};

template <typename T> class Tensor : public ITensor {
  public:
    // empty
    explicit Tensor(bool is_on_GPU = false) {
        if (is_on_GPU) {
            toGPU();
        }
    }

    // nchw
    Tensor(int n, int c, int h, int w, bool is_on_GPU = false) {
        dims_[0] = n;
        dims_[1] = c;
        dims_[2] = h;
        dims_[3] = w;
        size_ = sizeof(T) * n * c * h * w;
        if (is_on_GPU) {
            toGPU();
        }
    }

    // nchw in vector
    template <typename U>
    explicit Tensor(const std::vector<U> &dims, bool is_on_GPU = false) {
        memset(dims_, 0, sizeof(dims_));
        size_ = dims.empty() ? 0 : sizeof(T);
        int i = 0;
        for (auto d : dims) {
            size_ *= d;
            dims_[i++] = static_cast<int>(d);
        }
        if (is_on_GPU) {
            toGPU();
        }
    }

    const std::type_info &dtype() const override { return typeid(T); }

    void resize(int n, int c, int h, int w) {
        dims_[0] = n;
        dims_[1] = c;
        dims_[2] = h;
        dims_[3] = w;
        size_ = sizeof(T) * n * c * h * w;
        if (!is_on_GPU())
            data_.resize(n * c * h * w);
    }

    template <typename U> void resize(const std::vector<U> &dims) {
        memset(dims_, 0, sizeof(dims_));
        size_ = dims.empty() ? 0 : sizeof(T);
        int i = 0;
        for (auto d : dims) {
            size_ *= d;
            dims_[i++] = static_cast<int>(d);
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
            memset(dims_, 0, sizeof(dims_));
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
            if (dims_[2] == 1 && dims_[3] == 1) {
                // for weight kernel 1x1
                pack_ = true;
                transpose_ = false;
            }
            make_vkimg(vd, VK_IMAGE_USAGE_SAMPLED_BIT |
                               VK_IMAGE_USAGE_STORAGE_BIT |
                               VK_IMAGE_USAGE_TRANSFER_DST_BIT);
        }
        auto img = std::dynamic_pointer_cast<VulkanImage>(vkobj_);
#ifdef VK_EXT_host_image_copy
        if (vd->is_support_host_image_copy()) {
            img->hostImaggeTransition(VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        } else
#endif
        {
            if (cmd) {
                img->readBarrier(cmd->get());
            }
        }

        return img;
    }

    std::shared_ptr<VulkanImage>
    as_output_image(std::shared_ptr<VulkanDevice> &vd,
                    const std::shared_ptr<VulkanCommandBuffer> &cmd) {
        if (!vkobj_) {
            make_vkimg(vd, VK_IMAGE_USAGE_STORAGE_BIT |
                               VK_IMAGE_USAGE_SAMPLED_BIT |
                               VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
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
                img->writeBarrier(cmd->get());
            }
        }

        return img;
    }
    void copyToGPUImage(const std::shared_ptr<VulkanCommandPool> &cmdpool,
                        T *src = nullptr) {
        auto img = std::dynamic_pointer_cast<VulkanImage>(vkobj_);
        if (num_dims() < 3) {
            return;
        }
        auto dev = cmdpool->getVulkanDevice();
        VulkanCommandBuffer cmd(cmdpool, false);

#ifdef VK_EXT_host_image_copy
        if (dev->is_support_host_image_copy()) {
            std::vector<T> ptr(img->getImageSize());
            convertTensorToRGBA(ptr.data(), src);
            img->hostImageCopyToDevice(ptr.data());
            cmd.begin();
            img->readBarrier(cmd.get());
            cmd.end();
            cmd.submit(dev->getComputeQueue());
        } else
#endif
        {
            auto imagesize = img->getImageSize();
            auto stpool = cmdpool->getStagingBufferPool();
            uint64_t completed = cmdpool->getCompletedTimelineValue();
            stpool->reclaimCompleted(completed);
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
            auto submit_value = cmd.submit(dev->getComputeQueue());
            cmd.wait(dev->getComputeQueue());
            stpool->markSubmit(submit_value);
        }
        toGPU();
    }
    void copyToGPU(const std::shared_ptr<VulkanCommandPool> &cmdpool,
                   T *data = nullptr) {
        if (is_on_GPU()) {
            return;
        }
        if (vkobj_->getResourceType() == ResourceType::VK_IMAGE) {
            copyToGPUImage(cmdpool, data);
        } else {
            copyToGPUBuffer(cmdpool, data);
        }
        if (!data) {
            data_.clear();
            data_.shrink_to_fit();
        }
    }

    void copyToCPU(const std::shared_ptr<VulkanCommandPool> &cmdpool) {
        if (!is_on_GPU()) {
            printf("not on GPU\n");
            return;
        }
        data_.resize(num_elements());
        if (vkobj_->getResourceType() == ResourceType::VK_IMAGE) {
            copyImageToCPU(cmdpool);
        } else {
            copyBufferToCPU(cmdpool);
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
        VkFormat format = VK_FORMAT_R32G32B32A32_SFLOAT;
        if (sizeof(T) == 1) {
            format = VK_FORMAT_R8G8B8A8_UINT;
        } else if (sizeof(T) == 2) {
            format = VK_FORMAT_R16G16B16A16_SFLOAT;
        }
        auto batch = transpose_ ? dims_[1] : dims_[0];
        auto chan = transpose_ ? dims_[0] : dims_[1];
        uint32_t height = dims_[2];
        uint32_t width = dims_[3];
        uint32_t realheight = height * batch;
        auto chan4 = UP_DIV(chan, 4);
        if (pack_) {
            // one layer image, now for 1x1 kernel,
            // for NCHW, realW,realH (W * C4, N*H) and now (C4, N)
            // (Cout, Cin_per_group, kH, kW) now (Cin_per_group4, Cout)
            // after transpose it is (Cout4, Cin_per_group)
            width = width * chan4;
            chan4 = 1;
        }
        if (vd->is_support_host_image_copy()) {
#ifdef VK_EXT_host_image_copy
            flags |= VK_IMAGE_USAGE_HOST_TRANSFER_BIT;
#endif
        }

        auto vkdim = VkExtent3D{width, realheight, 1};
        vkobj_ = std::make_shared<VulkanImage>(vd, vkdim, chan4, flags, format);
    }

    void copyToGPUBuffer(const std::shared_ptr<VulkanCommandPool> &cmdpool,
                         T *src = nullptr) {
        auto buffer = std::dynamic_pointer_cast<VulkanBuffer>(vkobj_);

        auto dev = cmdpool->getVulkanDevice();
        VulkanCommandBuffer cmd(cmdpool, false);
        auto stpool = cmdpool->getStagingBufferPool();
        uint64_t completed = cmdpool->getCompletedTimelineValue();
        stpool->reclaimCompleted(completed);
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
        auto submit_value = cmd.submit(dev->getComputeQueue());
        cmd.wait(dev->getComputeQueue());
        stpool->markSubmit(submit_value);
        toGPU();
    }

    void copyBufferToCPU(const std::shared_ptr<VulkanCommandPool> &cmdpool) {
        auto buffer = std::dynamic_pointer_cast<VulkanBuffer>(vkobj_);

        auto dev = cmdpool->getVulkanDevice();
        VulkanCommandBuffer cmd(cmdpool, false);

        auto stpool = cmdpool->getStagingBufferPool();
        uint64_t completed = cmdpool->getCompletedTimelineValue();
        stpool->reclaimCompleted(completed);
        auto *buff = stpool->getBuffer();
        auto b = stpool->allocate(size_);
        if (!b) {
            printf("stpool alloc failed\n");
            return;
        }

        cmd.begin();
        buffer->copyBufferToStageBuffer(cmd.get(), buff, b->offset);
        cmd.end();
        auto submit_value = cmd.submit(dev->getComputeQueue());
        cmd.wait(dev->getComputeQueue());
        stpool->markSubmit(submit_value);
        std::memcpy(data_.data(), b->ptr, size_);
        toCPU();
    }
    void copyImageToCPU(const std::shared_ptr<VulkanCommandPool> &cmdpool) {
        auto img = std::dynamic_pointer_cast<VulkanImage>(vkobj_);

        auto dev = cmdpool->getVulkanDevice();

#ifdef VK_EXT_host_image_copy
        if (dev->is_support_host_image_copy()) {
            std::vector<T> rgba((img->getImageSize() / sizeof(T)));
            img->hostImageCopyToHost(rgba.data());
            convertRGBAToTensor(rgba.data());
        } else
#endif
        {
            VulkanCommandBuffer cmd(cmdpool, false);

            auto stpool = cmdpool->getStagingBufferPool();
            uint64_t completed = cmdpool->getCompletedTimelineValue();
            stpool->reclaimCompleted(completed);
            auto *buff = stpool->getBuffer();
            auto b = stpool->allocate(img->getImageSize());
            if (!b) {
                printf("stpool alloc failed\n");
                return;
            }

            cmd.begin();
            img->copyImageToBuffer(cmd.get(), buff, b->offset);
            cmd.end();
            auto submit_value = cmd.submit(dev->getComputeQueue());
            cmd.wait(dev->getComputeQueue());
            stpool->markSubmit(submit_value);
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
     * CPU packing for NCHW -> (W, H*N, arrayLayers=C4)
     */
    void convertTensorToRGBA(T *ptr, T *src = nullptr) {
        auto img = std::dynamic_pointer_cast<VulkanImage>(vkobj_);
        if (!img) {
            throw std::runtime_error(
                "Failed to cast VulkanResource to VulkanImage");
        }
        auto batch = transpose_ ? dims_[1] : dims_[0];
        auto chan = transpose_ ? dims_[0] : dims_[1];
        auto height = dims_[2];
        auto width = dims_[3];
        int chan4 = UP_DIV(chan, 4);

        const T *input = src ? src : data_.data();

        uint32_t row_pitch = width * 4 * sizeof(T);
        uint32_t layer_stride = batch * height * row_pitch;

        for (int c4 = 0; c4 < chan4; c4++) {
            for (int n = 0; n < batch; n++) {
                for (int h = 0; h < height; h++) {
                    for (int w = 0; w < width; w++) {
                        int dst_bytes_offset =
                            (pack_ ? ((n * height + h) * (width * chan4) +
                                      (w + c4 * width)) *
                                         4 * sizeof(T)
                                   : c4 * layer_stride +
                                         (n * height + h) * row_pitch +
                                         w * 4 * sizeof(T));
                        T *dst = reinterpret_cast<T *>(
                            reinterpret_cast<uint8_t *>(ptr) +
                            dst_bytes_offset);
                        for (int k = 0; k < 4; k++) {
                            int c = c4 * 4 + k;
                            if (c < chan) {
                                int offset =
                                    (transpose_
                                         ? (c * batch * height * width +
                                            n * height * width + h * width + w)
                                         : (n * chan * height * width +
                                            c * height * width + h * width +
                                            w));

                                dst[k] = input[offset];
                            } else {
                                dst[k] = T(0);
                            }
                        }
                    }
                }
            }
        }
    }

    void convertRGBAToTensor(T *ptr) {
        auto batch = transpose_ ? dims_[1] : dims_[0];
        auto chan = transpose_ ? dims_[0] : dims_[1];
        auto height = dims_[2];
        auto width = dims_[3];

        int chan4 = UP_DIV(chan, 4);

        uint32_t row_pitch = width * 4 * sizeof(T);
        uint32_t layer_stride = height * batch * row_pitch;
        T *src = nullptr;
        for (int c4 = 0; c4 < chan4; ++c4) {
            for (int n = 0; n < batch; n++) {
                for (int h = 0; h < height; h++) {
                    for (int w = 0; w < width; w++) {
                        int src_bytes_offset =
                            (pack_ ? ((n * height + h) * (width * chan4) +
                                      (w + c4 * width)) *
                                         4 * sizeof(T)
                                   : c4 * layer_stride +
                                         (n * height + h) * row_pitch +
                                         w * 4 * sizeof(T));
                        src = reinterpret_cast<T *>(
                            reinterpret_cast<uint8_t *>(ptr) +
                            src_bytes_offset);

                        for (int k = 0; k < 4; k++) {
                            int c = c4 * 4 + k;
                            if (c < chan) {
                                size_t offset;
                                if (transpose_) {
                                    offset = c * batch * height * width +
                                             n * height * width + h * width + w;
                                } else {
                                    offset = n * chan * height * width +
                                             c * height * width + h * width + w;
                                }
                                data_[offset] = src[k];
                            }
                        }
                    }
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
