// Copyright 2025 @junka
#ifndef CORE_TENSOR_HPP_
#define CORE_TENSOR_HPP_

#include <cstdint>
#include <iostream>
#include <memory>
#include <vector>

#include "vulkan/VulkanCommandBuffer.hpp"
#include "vulkan/VulkanDevice.hpp"
#include "vulkan/VulkanImage.hpp"
#include "vulkan/VulkanResource.hpp"

#define UP_DIV(x, y) (((x) + (y)-1) / (y))

namespace vkop {
namespace core {

template <typename T> class Tensor {
  public:
    // empty
    Tensor() = default;

    // nchw
    Tensor(int n, int c, int h, int w) : n_(n), c_(c), h_(h), w_(w) {
        ele_size_ = sizeof(T);
        fp16_ = (sizeof(T) == 2);
        size_ = ele_size_ * n_ * c_ * h_ * w_;
    }

    // nchw in vector
    explicit Tensor(const std::vector<int> &dims)
        : n_(dims[0]), c_(dims[1]), h_(dims[2]), w_(dims[3]) {
        ele_size_ = sizeof(T);
        fp16_ = (sizeof(T) == 2);
        size_ = ele_size_ * n_ * c_ * h_ * w_;
        data_.resize(n_ * c_ * h_ * w_);
    }

    // nchw in vector
    explicit Tensor(const std::vector<uint32_t> &dims)
        : n_(dims[0]), c_(dims[1]), h_(dims[2]), w_(dims[3]) {
        ele_size_ = sizeof(T);
        fp16_ = (sizeof(T) == 2);
        size_ = ele_size_ * n_ * c_ * h_ * w_;
        data_.resize(n_ * c_ * h_ * w_);
    }

    void resize(int n, int c, int h, int w) {
        n_ = n;
        c_ = c;
        h_ = h;
        w_ = w;
        ele_size_ = sizeof(T);
        fp16_ = (sizeof(T) == 2);
        size_ = ele_size_ * n_ * c_ * h_ * w_;
        data_.resize(n_ * c_ * h_ * w_);
    }

    void resize(const std::vector<int> &dims) {
        n_ = dims[0];
        c_ = dims[1];
        h_ = dims[2];
        w_ = dims[3];
        ele_size_ = sizeof(T);
        fp16_ = (sizeof(T) == 2);
        size_ = ele_size_ * n_ * c_ * h_ * w_;
        data_.resize(n_ * c_ * h_ * w_);
    }

    void printTensorShape() const {
        std::cout << "Tensor dimensions: [" << n_ << "," << c_ << "," << h_
                  << "," << w_ << "], "
                  << "size: " << sizeof(T) << std::endl;
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

    std::vector<int> getTensorShape() {
        return std::vector<int>{n_, c_, h_, w_};
    }

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
    std::vector<T> convertTensorToRGBA() {
        auto obj = vkobj_.lock();
        assert(obj->getResourceType() == ResourceType::VK_IMAGE);
        auto img = std::dynamic_pointer_cast<VulkanImage>(obj);
        if (!img) {
            throw std::runtime_error(
                "Failed to cast VulkanResource to VulkanImage");
        }
        auto batch = n_;
        auto depth = c_;
        auto height = h_;
        auto width = w_;

        int stride_w = 1;
        int stride_h = width;
        int stride_c = width * height;
        int stride_n = width * height * depth;
        int realdepth = UP_DIV(depth, 4);
        int realwidth = width * UP_DIV(depth, 4);
        int realheight = height * batch;

        std::vector<T> tmp(realheight * realwidth * img->getImageChannelNum() *
                           img->getImageChannelSize());
        T *ptr = tmp.data();
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
        return tmp;
    }

    void convertRGBAToTensor(T *ptr) {
        auto batch = n_;
        auto depth = c_;
        auto height = h_;
        auto width = w_;

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

    std::shared_ptr<VulkanImage> make_vkimg(std::shared_ptr<VulkanDevice> &vd,
                                            uint32_t flags) {
        auto vkimg = std::make_shared<VulkanImage>(
            vd,
            VkExtent3D{static_cast<uint32_t>(w_ * UP_DIV(c_, 4)),
                       static_cast<uint32_t>(h_ * n_), 1},
            flags, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            (fp16_ ? VK_FORMAT_R16G16B16A16_SFLOAT
                   : VK_FORMAT_R32G32B32A32_SFLOAT));

        vkobj_ = vkimg;
        return vkimg;
    }

  private:
    int n_;
    int c_;
    int h_;
    int w_;

    int ele_size_;
    int size_;

    bool fp16_;

    std::vector<T> data_;
    std::weak_ptr<VulkanResource> vkobj_;
};

} // namespace core
} // namespace vkop

#endif // CORE_TENSOR_HPP_
