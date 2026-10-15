#include "vulkan/VulkanDevice.hpp"
#include "vulkan/VulkanInstance.hpp"
#include "include/logger.hpp"
#include "core/Tensor.hpp"
#include "core/runtime.hpp"

#include <cstdio>
#include <memory>
#include <string>
#include <vector>

using vkop::VulkanInstance;
using vkop::VulkanDevice;
using vkop::core::ITensor;
using vkop::core::Runtime;

namespace {

// The converter unifies every float/fp16 initializer with <= 2 dims, so for
// this fixture that is conv.bias and conv.weight_scale; conv.weight is 4-D and
// stays a per-tensor image.
const char *kUnifiedNames[] = {"conv.bias", "conv.weight_scale"};

bool Load(const std::shared_ptr<vkop::VulkanCommandPool> &cmdpool,
          const std::string &file, std::shared_ptr<Runtime> &rt) {
    try {
        rt = std::make_shared<Runtime>(cmdpool, TEST_DATA_PATH"/" + file, 0);
        // The unified region is sub-allocated as uniform BufferViews, so this
        // is the backend -u artifacts are loaded on.
        rt->set_backend_buffer(true);
        rt->LoadModel();
    } catch (const std::exception &e) {
        printf("FAIL: loading %s threw: %s\n", file.c_str(), e.what());
        return false;
    }
    return true;
}

bool Fetch(const std::shared_ptr<Runtime> &rt,
           const std::shared_ptr<vkop::VulkanCommandPool> &cmdpool,
           const char *name, std::vector<float> &out) {
    auto t = vkop::core::as_tensor<float>(rt->GetInitializer(name));
    if (!t) {
        printf("FAIL: no %s initializer\n", name);
        return false;
    }
    t->copyToCPU(cmdpool);
    out.assign(t->num_elements(), 0.0F);
    for (int i = 0; i < t->num_elements(); ++i) {
        out[i] = t->at(i);
    }
    return true;
}

} // namespace

// -u regression: LoadModel reads each unified member's dtype/dims out of its
// InitializerEntry side-table row before sub-allocating it from the shared
// region (core/runtime.cpp:419). Dropping those rows on the writer side made
// the lookup return a default-constructed Initializer and load throw
// "vkop: unified initializer  has unrecognized dtype ''".
//
// Both fixtures are converted from the same add_conv_model.onnx, so comparing
// the unified artifact's members against the per-tensor artifact's also checks
// the sub-allocation offsets: a wrong offset hands back the neighbouring
// tensor's bytes rather than throwing.
int main() {
    Logger::getInstance().setLevel(LOG_INFO);
    const auto &phydevs = VulkanInstance::getVulkanInstance().getPhysicalDevices();
    auto dev = std::make_shared<VulkanDevice>(phydevs[0]);
    if (dev->getDeviceName().find("llvmpipe") != std::string::npos) {
        printf("no valid vulkan device\n");
        return -1;
    }
    auto cmdpool = std::make_shared<vkop::VulkanCommandPool>(dev);

    std::shared_ptr<Runtime> per_tensor, unified;
    if (!Load(cmdpool, "add_conv_model.vkopbin", per_tensor)) return 1;
    if (!Load(cmdpool, "add_conv_model_unified.vkopbin", unified)) return 1;

    for (const char *name : kUnifiedNames) {
        std::vector<float> expected, got;
        if (!Fetch(per_tensor, cmdpool, name, expected)) return 1;
        if (!Fetch(unified, cmdpool, name, got)) return 1;
        if (expected.size() != got.size()) {
            printf("FAIL: %s has %zu elements unified vs %zu per-tensor\n", name,
                   got.size(), expected.size());
            return 1;
        }
        for (size_t i = 0; i < expected.size(); ++i) {
            if (expected[i] != got[i]) {
                printf("FAIL: %s[%zu] is %.9g unified vs %.9g per-tensor\n", name,
                       i, got[i], expected[i]);
                return 1;
            }
        }
        printf("ok: %s (%zu elements) identical through the unified region\n",
               name, got.size());
    }
    return 0;
}
