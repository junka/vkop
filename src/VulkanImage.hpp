#ifndef VULKAN_IMAGE_HPP
#define VULKAN_IMAGE_HPP

#include "vulkan/vulkan.hpp"

#include <cstdint>
#include <variant>
#include <memory>

#include "VulkanResource.hpp"
#include "VulkanBuffer.hpp"

#define UP_DIV(x, y) (((x) + (y) - 1) / (y))

namespace vkop {

// VulkanImage class inheriting from VulkanResource
class VulkanImage : public VulkanResource {
public:
    VulkanImage(VkPhysicalDevice physicalDevice, const uint32_t queueFamilyIndex, VkDevice device,
        VkExtent3D dim,  VkFormat format, VkImageUsageFlags usage, VkMemoryPropertyFlags requireProperties);
    ~VulkanImage();

    ResourceType getResourceType() const override {
        return ResourceType::Image;
    }

    std::variant<VkDescriptorImageInfo, VkDescriptorBufferInfo> getDescriptorInfo() const override;

    void transitionImageLayout(VkCommandBuffer commandBuffer, VkImageLayout newLayout,
                VkAccessFlags dstAccessMask, VkPipelineStageFlags sourceStage, VkPipelineStageFlags destinationStage);

    void transferBarrier(VkCommandBuffer commandBuffer, VkImageLayout newLayout,
        VkAccessFlags dstAccessMask);
    void transferReadBarrier(VkCommandBuffer commandBuffer);
    void transferWriteBarrier(VkCommandBuffer commandBuffer);

    void readBarrier(VkCommandBuffer commandBuffer);

    void writeBarrier(VkCommandBuffer commandBuffer);
    

    void copyImageToBuffer(VkCommandBuffer commandBuffer, VulkanBuffer& buffer);
    void copyBufferToImage(VkCommandBuffer commandBuffer, VulkanBuffer& buffer);

    void hostImageCopyToDevice(void *ptr);
    void hostImageCopyToHost(void *ptr);

    void stagingBufferCopyToImage(VkCommandBuffer commandBuffer, void *ptr);
    void stagingBufferCopyToHost(VkCommandBuffer commandBuffer, void *ptr);

    void hostImaggeTransition(VkImageLayout newLayout);

    template<typename T>
    std::vector<T> convertNCHWToRGBA(const T *data, std::vector<int> nchw)
    {
        int batch = nchw[0];
        int depth = nchw[1];
        int height = nchw[2];
        int width = nchw[3];
    
        int stride_w = 1;
        int stride_h = width;
        int stride_c = width * height;
        int stride_n = width * height * depth;
        int realdepth = UP_DIV(depth, 4);
        int realwidth = width * UP_DIV(depth, 4);
        int realheight = height * batch;
    
        std::vector<T> tmp(realheight * realwidth * getImageChannelNum() * getImageChannelSize());
        T *ptr = tmp.data();
        uint32_t rowPitch = realwidth * getImageChannelNum() * getImageChannelSize();
        T *dst = reinterpret_cast<T *>(ptr);
        for (int b = 0; b < batch; b++) {
            T* batchstart = reinterpret_cast<T *>(reinterpret_cast<uint8_t *>(ptr) + b * height * rowPitch);
            for (int c = 0; c < realdepth; c++) {
                dst = reinterpret_cast<T *>(batchstart) + c * 4 * width;
                for (int h = 0; h < height; h++) {
                    for (int w = 0; w < width; w++) {
                        int offset = b * stride_n + 4 * c * stride_c + h * stride_h + w * stride_w;
    
                        dst[w * 4 + 0] = data[offset];
                        dst[w * 4 + 1] = (4 * c + 1 < depth) ? data[stride_c + offset] : 0.0f;
                        dst[w * 4 + 2] = (4 * c + 2 < depth) ? data[2 * stride_c + offset] : 0.0f;
                        dst[w * 4 + 3] = (4 * c + 3 < depth) ? data[3 * stride_c + offset] : 0.0f;
                    }
                    dst = reinterpret_cast<T *>(reinterpret_cast<uint8_t *>(dst) + rowPitch);
                }
            }
        }
        return tmp;
    }
    template<typename T>
    std::vector<T> convertRGBAToNCHW(T *ptr, std::vector<int> nchw)
    {
        int batch = nchw[0];
        int depth = nchw[1];
        int height = nchw[2];
        int width = nchw[3];
    
        int stride_w = 1;
        int stride_h = width;
        int stride_c = width * height;
        int stride_n = width * height * depth;

        int realdepth = UP_DIV(depth, 4);
        int realwidth = width * UP_DIV(depth, 4);
        
        std::vector<T> retdata(batch * height * depth * width);
        T *data = retdata.data();

        uint32_t rowPitch = realwidth * 4 * sizeof(T);
        T *dst;
        for (int b = 0; b < batch; b++) {
            T* batchstart = reinterpret_cast<T *>(reinterpret_cast<uint8_t *>(ptr) + b * height * rowPitch);
            for (int c = 0; c < realdepth; c++) {
                dst = reinterpret_cast<T *>(batchstart) + c * width * 4;
                for (int h = 0; h < height; h++) {
                    for (int w = 0; w < width; w++) {
                        int offset = b * stride_n + 4 * c * stride_c + h * stride_h + w * stride_w;
                        data[offset] = dst[w * 4 + 0];
                        if (4 * c + 1 < depth) {
                            data[stride_c + offset] = dst[w * 4 + 1];
                        }
                        if (4 * c + 2 < depth) {
                            data[stride_c * 2 + offset] = dst[w * 4 + 2];
                        }
                        if (4 * c + 3 < depth) {
                            data[stride_c * 3 + offset] = dst[w * 4 + 3];
                        }
                    }
                    dst = reinterpret_cast<T *>(reinterpret_cast<uint8_t *>(dst) + rowPitch);
                }
            }
        }
        return retdata;
    }
private:
    VkExtent3D m_dim;
    VkFormat m_format;
    VkImageUsageFlags m_usage;
    VkImageType m_imagetype;
    VkImageLayout m_layout;
    VkAccessFlags m_access;

    VkImage m_image;
    VkImageView m_imageView;
    VkSampler m_sampler;

    int m_chansize;
    int m_chans;

    std::unique_ptr<VulkanBuffer> m_stagingBuffer;

    void calcImageSize();
    int getImageSize() { return m_chansize * m_chans * m_dim.width * m_dim.height * m_dim.depth;}

    int getImageChannelSize() { return m_chansize; }
    int getImageChannelNum() { return m_chans; }
    int getImageWidth() { return m_dim.width; }
    int getImageHeight() { return m_dim.height; }
    int getImageDepth() { return m_dim.depth; }

    void createImage();

    void createImageView();

    void createSampler();

    void destroyImage();

    void createStagingBuffer(bool writeonly);
};
}
#endif