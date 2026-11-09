// Copyright 2026 @junka
// Buffer-device-address plumbing probe.
//
// This is the compatibility contract in executable form:
//   - on a device without BDA the test SKIPs (every op must stay on the
//     descriptor path, and nothing here may be required for a model to run);
//   - on a supporting device an ordinary VulkanBuffer exposes a nonzero,
//     stable address and sub-range views are root_address + offset;
//   - host-imported buffers are deliberately NOT addressable in phase 1, so a
//     driver that rejects VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT on imported
//     memory cannot take the per-step input path down with it.
//
// Run: ./BufferDeviceAddressTest
// VKOP_NO_BDA=1 exercises the fallback: the test skips, and the rest of the
// suite must behave exactly as before.

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <stdexcept>

#include "include/logger.hpp"
#include "vulkan/VulkanBuffer.hpp"
#include "vulkan/VulkanDevice.hpp"
#include "vulkan/VulkanInstance.hpp"

using vkop::VulkanBuffer;
using vkop::VulkanDevice;
using vkop::VulkanInstance;

namespace {

int failures = 0;

void check(bool condition, const char *what) {
    if (!condition) {
        std::printf("FAIL: %s\n", what);
        failures++;
    }
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
    if (!dev->is_support_buffer_device_address()) {
        std::printf("SKIP: buffer device address unsupported (descriptor path "
                    "only); set VKOP_NO_BDA only to exercise this branch\n");
        return 0;
    }

    // Owning buffer: the one path whose allocation is covered by the
    // device-init probe (ordinary memory through VMA or raw vkAllocateMemory).
    auto base = std::make_shared<VulkanBuffer>(
        dev, 4096,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    check(base->hasDeviceAddress(), "owning buffer should be BDA-capable");
    const VkDeviceAddress base_addr = base->getDeviceAddress();
    check(base_addr != 0, "base address is nonzero");
    check(base->getDeviceAddress() == base_addr,
          "address is cached and stable across calls");
    std::printf("base address 0x%llx (mod 16 = %llu, mod 8 = %llu)\n",
                static_cast<unsigned long long>(base_addr),
                static_cast<unsigned long long>(base_addr % 16),
                static_cast<unsigned long long>(base_addr % 8));

    // Borrowed sub-range view: descriptor path binds parent+offset, BDA path
    // must report the same byte window as an absolute address.
    auto view = std::make_shared<VulkanBuffer>(base, 256, 512);
    check(view->hasDeviceAddress(), "view forwards the parent capability");
    check(view->getDeviceAddress() == base_addr + 256,
          "view address = root + range_offset");

    // View of a view: abs_offset accumulates every ancestor.
    auto nested = std::make_shared<VulkanBuffer>(view, 128, 128);
    check(nested->getDeviceAddress() == base_addr + 384,
          "nested view address accumulates ancestor offsets");

    // Host-imported memory stays descriptor-only. Creating the import must
    // still succeed while BDA is on; only the address is unavailable.
    if (dev->is_support_host_import()) {
        const size_t align = dev->host_import_alignment();
        void *host = nullptr;
        if (posix_memalign(&host, align, align * 2) != 0) {
            std::printf("FAIL: posix_memalign for host-import probe\n");
            failures++;
        } else {
            try {
                auto imported = std::make_shared<VulkanBuffer>(
                    dev, 64,
                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                        VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                    host, align * 2);
                check(!imported->hasDeviceAddress(),
                      "host-imported buffer is not BDA-capable");
                bool threw = false;
                try {
                    (void)imported->getDeviceAddress();
                } catch (const std::runtime_error &) {
                    threw = true;
                }
                check(threw,
                      "getDeviceAddress() throws for a host-imported buffer");
                imported.reset(); // release the VkDeviceMemory wrapper
            } catch (const std::runtime_error &e) {
                std::printf("FAIL: host-import buffer creation failed with BDA "
                            "enabled: %s\n",
                            e.what());
                failures++;
            }
            std::free(host);
        }
    } else {
        std::printf(
            "SKIP: host import unsupported; import+BDA branch untested\n");
    }

    std::printf(failures ? "RESULT: %d FAILURE(S)\n" : "RESULT: all pass\n",
                failures);
    return failures == 0 ? 0 : 1;
}
