// Copyright 2026 @junka
// VK_EXT_descriptor_buffer host-side cost probe.
//
// Answers three concrete questions on the machine it runs on:
//   1. how expensive is vkGetDescriptorEXT compared with vkUpdateDescriptorSets
//      for the same set of bindings?
//   2. how expensive is recording vkCmdSetDescriptorBufferOffsetsEXT compared
//      with vkCmdBindDescriptorSets?
//   3. how much does the per-dispatch vkCmdBindDescriptorBuffersEXT (what the
//      engine's per-pipeline descriptor buffer does today) add on top of the
//      offset command alone?
//
// Skips cleanly when the device cannot use descriptor buffers. Not a gtest:
// capability-dependent skip is a normal outcome.
//
// Build/run (dev box): ./build/tests/DescriptorBufferProbe
// b3 (no CMake tests): g++-13 -O2 -std=gnu++17 -I<repo> -isystem
//   <repo>/model/generated DescriptorBufferProbe.cpp
//   <repo>/build_llm/libvkop.a -ldl -lpthread -o DescriptorBufferProbe

#include <vulkan/vulkan.h>

#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

#include "include/logger.hpp"
#include "vulkan/VulkanBuffer.hpp"
#include "vulkan/VulkanDevice.hpp"
#include "vulkan/VulkanInstance.hpp"
#include "vulkan/VulkanLib.hpp"

using vkop::VulkanBuffer;
using vkop::VulkanDevice;
using vkop::VulkanInstance;

namespace {

constexpr uint32_t kBindings = 7;
int failures = 0;

void check(bool cond, const char *what) {
    if (!cond) {
        std::printf("FAIL: %s\n", what);
        failures++;
    }
}

uint64_t now_ns() {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch())
            .count());
}

std::vector<VkDescriptorSetLayoutBinding> make_bindings() {
    std::vector<VkDescriptorSetLayoutBinding> b(kBindings);
    for (uint32_t i = 0; i < kBindings; ++i) {
        b[i].binding = i;
        b[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        b[i].descriptorCount = 1;
        b[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    return b;
}

} // namespace

int main() {
    Logger::getInstance().setLevel(LOG_INFO);
    const auto &phydevs =
        VulkanInstance::getVulkanInstance().getPhysicalDevices();
    if (phydevs.empty()) {
        std::printf("SKIP: no Vulkan device\n");
        return 0;
    }
    auto dev = std::make_shared<VulkanDevice>(phydevs[0]);
    std::printf("GPU: %s\n", dev->getDeviceName().c_str());
    const vkop::DescriptorBufferInfo *dbi = dev->descriptor_buffer_info();
    if (!dbi) {
        std::printf("SKIP: VK_EXT_descriptor_buffer unsupported\n");
        return 0;
    }
    VkDevice vkdev = dev->getLogicalDevice();

    VkPhysicalDeviceMemoryProperties mp{};
    vkop::vkGetPhysicalDeviceMemoryProperties(dev->getPhysicalDevice(), &mp);
    std::printf("memory types (heap MiB, flags):\n");
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i) {
        const auto &t = mp.memoryTypes[i];
        std::printf("  [%u] heap=%u %llu MiB flags=0x%x%s%s%s\n", i,
                    t.heapIndex,
                    (unsigned long long)(mp.memoryHeaps[t.heapIndex].size >> 20),
                    (unsigned)t.propertyFlags,
                    (t.propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
                        ? " DEVICE_LOCAL"
                        : "",
                    (t.propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
                        ? " HOST_VISIBLE"
                        : "",
                    (t.propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
                        ? " HOST_COHERENT"
                        : "");
    }

    // One real storage buffer all bindings point at.
    auto buf = std::make_shared<VulkanBuffer>(
        dev, 1u << 20,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    const VkDeviceAddress buf_addr = buf->getDeviceAddress();

    // Descriptor-buffer layout.
    auto bindings = make_bindings();
    VkDescriptorSetLayoutCreateInfo dbl{};
    dbl.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dbl.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_DESCRIPTOR_BUFFER_BIT_EXT;
    dbl.bindingCount = kBindings;
    dbl.pBindings = bindings.data();
    VkDescriptorSetLayout db_layout = VK_NULL_HANDLE;
    check(vkop::vkCreateDescriptorSetLayout(vkdev, &dbl, nullptr, &db_layout) ==
              VK_SUCCESS,
          "create descriptor-buffer layout");

    // Plain layout for the vkUpdateDescriptorSets arm.
    VkDescriptorSetLayoutCreateInfo pl{};
    pl.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    pl.bindingCount = kBindings;
    pl.pBindings = bindings.data();
    VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
    check(vkop::vkCreateDescriptorSetLayout(vkdev, &pl, nullptr, &set_layout) ==
              VK_SUCCESS,
          "create plain layout");

    VkPipelineLayoutCreateInfo plc{};
    plc.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    VkPipelineLayout db_pl = VK_NULL_HANDLE, set_pl = VK_NULL_HANDLE;
    plc.setLayoutCount = 1;
    plc.pSetLayouts = &db_layout;
    check(vkop::vkCreatePipelineLayout(vkdev, &plc, nullptr, &db_pl) == VK_SUCCESS,
          "create db pipeline layout");
    plc.pSetLayouts = &set_layout;
    check(vkop::vkCreatePipelineLayout(vkdev, &plc, nullptr, &set_pl) == VK_SUCCESS,
          "create set pipeline layout");

    VkDeviceSize layout_size = 0;
    dbi->getLayoutSize(vkdev, db_layout, &layout_size);
    std::vector<VkDeviceSize> binding_off(kBindings);
    for (uint32_t i = 0; i < kBindings; ++i) {
        dbi->getBindingOffset(vkdev, db_layout, i, &binding_off[i]);
    }
    std::printf("layout: size=%llu offsets:", (unsigned long long)layout_size);
    for (auto o : binding_off) {
        std::printf(" %llu", (unsigned long long)o);
    }
    std::printf("\n");

    // 256 slots of descriptor memory so the writes are not all the same address.
    constexpr uint32_t kSlots = 256;
    const uint32_t stride =
        static_cast<uint32_t>((layout_size + dbi->offsetAlignment - 1) /
                              dbi->offsetAlignment * dbi->offsetAlignment);
    void *desc_mem = nullptr;
    check(posix_memalign(&desc_mem, 4096, size_t(stride) * kSlots) == 0,
          "descriptor host memory");

    // Bindings for both write APIs.
    std::vector<VkDescriptorBufferInfo> buffer_infos(kBindings);
    std::vector<VkDescriptorAddressInfoEXT> addr_infos(kBindings);
    std::vector<VkDescriptorGetInfoEXT> get_infos(kBindings);
    std::vector<VkWriteDescriptorSet> writes(kBindings);
    for (uint32_t i = 0; i < kBindings; ++i) {
        buffer_infos[i].buffer = buf->getBuffer();
        buffer_infos[i].offset = 0;
        buffer_infos[i].range = 1u << 20;

        addr_infos[i].sType = VK_STRUCTURE_TYPE_DESCRIPTOR_ADDRESS_INFO_EXT;
        addr_infos[i].address = buf_addr;
        addr_infos[i].range = 1u << 20;
        addr_infos[i].format = VK_FORMAT_UNDEFINED;

        get_infos[i].sType = VK_STRUCTURE_TYPE_DESCRIPTOR_GET_INFO_EXT;
        get_infos[i].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        get_infos[i].data.pStorageBuffer = &addr_infos[i];

        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstBinding = i;
        writes[i].dstArrayElement = 0;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &buffer_infos[i];
    }

    // ---- 1. vkGetDescriptorEXT vs vkUpdateDescriptorSets -------------------
    constexpr uint32_t kIter = 100000;
    for (uint32_t i = 0; i < 1000; ++i) {
        dbi->getDescriptor(vkdev, &get_infos[0], dbi->storageBufferSize,
                           static_cast<char *>(desc_mem) +
                               (i % kSlots) * stride + binding_off[0]);
    }
    uint64_t t0 = now_ns();
    for (uint32_t i = 0; i < kIter; ++i) {
        char *slot = static_cast<char *>(desc_mem) + (i % kSlots) * stride;
        for (uint32_t b = 0; b < kBindings; ++b) {
            dbi->getDescriptor(vkdev, &get_infos[b], dbi->storageBufferSize,
                               slot + binding_off[b]);
        }
    }
    uint64_t get_ns = now_ns() - t0;

    VkDescriptorPoolSize ps{};
    ps.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    ps.descriptorCount = kBindings;
    VkDescriptorPoolCreateInfo pci{};
    pci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pci.maxSets = 1;
    pci.poolSizeCount = 1;
    pci.pPoolSizes = &ps;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    check(vkop::vkCreateDescriptorPool(vkdev, &pci, nullptr, &pool) == VK_SUCCESS,
          "pool");
    VkDescriptorSetAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    ai.descriptorPool = pool;
    ai.descriptorSetCount = 1;
    ai.pSetLayouts = &set_layout;
    VkDescriptorSet set = VK_NULL_HANDLE;
    check(vkop::vkAllocateDescriptorSets(vkdev, &ai, &set) == VK_SUCCESS, "set");
    for (auto &w : writes) {
        w.dstSet = set;
    }
    t0 = now_ns();
    for (uint32_t i = 0; i < kIter; ++i) {
        vkop::vkUpdateDescriptorSets(vkdev, kBindings, writes.data(), 0, nullptr);
    }
    uint64_t upd_ns = now_ns() - t0;

    std::printf(
        "vkGetDescriptorEXT      : %8.1f ns/call (%u calls/iter)\n",
        double(get_ns) / (double(kIter) * kBindings), kBindings);
    std::printf("vkUpdateDescriptorSets  : %8.1f ns/call (%u writes/call)\n",
                double(upd_ns) / double(kIter), kBindings);

    // ---- 2/3. command recording costs -------------------------------------
    // A real descriptor buffer to bind.
    auto desc_buf = std::make_shared<VulkanBuffer>(
        dev, size_t(stride) * kSlots,
        VK_BUFFER_USAGE_RESOURCE_DESCRIPTOR_BUFFER_BIT_EXT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VkDescriptorBufferBindingInfoEXT binding{};
    binding.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_BUFFER_BINDING_INFO_EXT;
    binding.address = desc_buf->getDeviceAddress();
    binding.usage = VK_BUFFER_USAGE_RESOURCE_DESCRIPTOR_BUFFER_BIT_EXT;

    VkCommandPoolCreateInfo cpi{};
    cpi.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cpi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    VkCommandPool cmdpool = VK_NULL_HANDLE;
    check(vkop::vkCreateCommandPool(vkdev, &cpi, nullptr, &cmdpool) == VK_SUCCESS,
          "cmd pool");
    VkCommandBufferAllocateInfo cai{};
    cai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cai.commandPool = cmdpool;
    cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cai.commandBufferCount = 1;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    check(vkop::vkAllocateCommandBuffers(vkdev, &cai, &cmd) == VK_SUCCESS, "cmd");

    constexpr uint32_t kCmdIter = 20000;
    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    const uint32_t buffer_index = 0;
    VkDeviceSize offset = 0;

    // 2a. bind sets
    vkop::vkResetCommandBuffer(cmd, 0);
    vkop::vkBeginCommandBuffer(cmd, &bi);
    t0 = now_ns();
    for (uint32_t i = 0; i < kCmdIter; ++i) {
        vkop::vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, set_pl, 0,
                                1, &set, 0, nullptr);
    }
    uint64_t bind_set_ns = now_ns() - t0;
    vkop::vkEndCommandBuffer(cmd);

    // 2b. offsets only (buffer bound once)
    vkop::vkResetCommandBuffer(cmd, 0);
    vkop::vkBeginCommandBuffer(cmd, &bi);
    dbi->cmdBindBuffers(cmd, 1, &binding);
    t0 = now_ns();
    for (uint32_t i = 0; i < kCmdIter; ++i) {
        offset = (i % kSlots) * stride;
        dbi->cmdSetOffsets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, db_pl, 0, 1,
                           &buffer_index, &offset);
    }
    uint64_t off_ns = now_ns() - t0;
    vkop::vkEndCommandBuffer(cmd);

    // 3. per-dispatch buffer rebind + offsets (today's engine behaviour)
    vkop::vkResetCommandBuffer(cmd, 0);
    vkop::vkBeginCommandBuffer(cmd, &bi);
    t0 = now_ns();
    for (uint32_t i = 0; i < kCmdIter; ++i) {
        dbi->cmdBindBuffers(cmd, 1, &binding);
        offset = (i % kSlots) * stride;
        dbi->cmdSetOffsets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, db_pl, 0, 1,
                           &buffer_index, &offset);
    }
    uint64_t rebind_ns = now_ns() - t0;
    vkop::vkEndCommandBuffer(cmd);

    std::printf("vkCmdBindDescriptorSets    : %8.1f ns/cmd\n",
                double(bind_set_ns) / kCmdIter);
    std::printf("vkCmdSetDescriptorBufferOffsets (buffer pre-bound): %8.1f "
                "ns/cmd\n",
                double(off_ns) / kCmdIter);
    std::printf("bind buffer + set offsets  : %8.1f ns/cmd\n",
                double(rebind_ns) / kCmdIter);

    std::free(desc_mem);
    std::printf("%s\n", failures ? "RESULT: FAIL" : "RESULT: OK");
    std::fflush(nullptr);
    // b3's NVIDIA EGL atexit handler SIGSEGVs after main; _exit reports the
    // real status and skips the teardown that never mattered for a probe.
    _exit(failures ? 1 : 0);
}
