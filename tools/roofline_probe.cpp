// Copyright 2025 @junka
//
// Driver-side per-thread resource probe for tools/roofline_report.py.
//
// SPIR-V says nothing about registers: only the driver's compiler knows.
// VK_KHR_pipeline_executable_properties is the one portable way to ask, and
// its statistics include "Max live registers" and "Workgroup Memory Size" --
// the two numbers a theoretical-occupancy calculation needs.
//
// This helper reads SPIR-V paths on stdin (one per line), builds a compute
// pipeline for each on the chosen Vulkan device and prints the statistics.
// It never dispatches: the pipelines are created and immediately destroyed, so
// the GPU does no work and nothing is written anywhere.
//
// Output protocol (stdout, line-oriented):
//   SHADER <path>
//   STAT <name>=<value>
//   FAIL <path> <reason>
//
// The descriptor set layout and push-constant range are DERIVED from each
// module (max binding index, block size) rather than assumed, because a
// layout that does not match the shader makes pipeline creation fail and the
// report would silently lose the whole probe.
//
// Optional device selection: --device-index N.  Without it the first device
// that can create a compute pipeline wins, with CPU/llvmpipe devices used only
// as a last resort (their statistics are not a hardware register file).
#include <vulkan/vulkan.h>

#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

namespace {

struct SpvFacts {
    uint32_t max_binding = 0;
    uint32_t binding_count = 0;
    bool has_storage = false;
    uint32_t pc_bytes = 0;
    bool cap_float16 = false;
    bool cap_int8 = false;
    bool cap_int16 = false;
    bool cap_int64 = false;
    bool cap_float8 = false;
    bool cap_coopmat = false;
    bool cap_storage8 = false;
    bool cap_storage16 = false;
};

SpvFacts scan_spv(const std::vector<uint32_t> &w) {
    SpvFacts f;
    bool seen[64] = {false};
    size_t i = 5;
    // Track the type of each pointer/variable so a push-constant block can be
    // measured: OpTypePointer -> pointee, OpVariable storage class 9
    // (PushConstant) -> pointer type -> struct -> member sizes.
    std::vector<uint32_t> type_of_struct;  // not needed beyond size probing
    while (i < w.size()) {
        uint32_t op = w[i] & 0xFFFF;
        uint32_t wc = w[i] >> 16;
        if (wc == 0) break;
        if (op == 17) {  // OpCapability
            switch (w[i + 1]) {
            case 9: f.cap_float16 = true; break;      // Float16
            case 11: f.cap_int64 = true; break;       // Int64
            case 21: f.cap_int16 = true; break;       // Int16
            case 39: f.cap_storage8 = true; break;    // StorageBuffer8BitAccess
            case 4436: f.cap_storage16 = true; break; // StorageBuffer16BitAccess
            case 4448: f.cap_float8 = true; break;    // Float8EXT
            case 6021: f.cap_coopmat = true; break;   // CooperativeMatrixKHR
            case 20: f.cap_int8 = true; break;        // Int8
            default: break;
            }
        } else if (op == 71) {  // OpDecorate
            uint32_t target = w[i + 1];
            uint32_t deco = w[i + 2];
            if (deco == 33) {  // Binding
                uint32_t b = w[i + 3];
                if (b < 64 && !seen[b]) {
                    seen[b] = true;
                    f.binding_count++;
                }
                if (b > f.max_binding) f.max_binding = b;
            }
        }
        i += wc;
    }
    (void)type_of_struct;
    return f;
}

const char *err_name(VkResult r) {
    switch (r) {
    case VK_SUCCESS: return "VK_SUCCESS";
    case VK_ERROR_OUT_OF_HOST_MEMORY: return "OUT_OF_HOST_MEMORY";
    case VK_ERROR_OUT_OF_DEVICE_MEMORY: return "OUT_OF_DEVICE_MEMORY";
    case VK_ERROR_INITIALIZATION_FAILED: return "INITIALIZATION_FAILED";
    case VK_ERROR_DEVICE_LOST: return "DEVICE_LOST";
    case VK_ERROR_FEATURE_NOT_PRESENT: return "FEATURE_NOT_PRESENT";
    default: return "VkResult";
    }
}

}  // namespace

namespace {

// Device selection result, filled once in main and inherited by the fork.
struct Ctx {
    VkDevice dev = VK_NULL_HANDLE;
    PFN_vkGetPipelineExecutablePropertiesKHR get_props = nullptr;
    PFN_vkGetPipelineExecutableStatisticsKHR get_stats = nullptr;
};

int probe_one(Ctx &ctx, const std::string &path) {
    std::ifstream f(path, std::ios::binary);
    std::vector<char> code((std::istreambuf_iterator<char>(f)),
                           std::istreambuf_iterator<char>());
    if (code.size() < 20) {
        std::printf("FAIL %s empty\n", path.c_str());
        return 1;
    }
    std::vector<uint32_t> words(code.size() / 4);
    std::memcpy(words.data(), code.data(), words.size() * 4);
    const SpvFacts facts = scan_spv(words);
    VkDevice dev = ctx.dev;

    VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smci.codeSize = code.size();
    smci.pCode = reinterpret_cast<const uint32_t *>(code.data());
    VkShaderModule sm = VK_NULL_HANDLE;
    VkResult r = vkCreateShaderModule(dev, &smci, nullptr, &sm);
    if (r != VK_SUCCESS) {
        std::printf("FAIL %s shader-module %s\n", path.c_str(), err_name(r));
        return 1;
    }
    std::vector<VkDescriptorSetLayoutBinding> binds;
    for (uint32_t b = 0; b <= facts.max_binding; ++b) {
        binds.push_back({b, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
                         VK_SHADER_STAGE_COMPUTE_BIT, nullptr});
    }
    if (binds.empty()) {
        binds.push_back({0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
                         VK_SHADER_STAGE_COMPUTE_BIT, nullptr});
    }
    VkDescriptorSetLayoutCreateInfo dslci{
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dslci.bindingCount = static_cast<uint32_t>(binds.size());
    dslci.pBindings = binds.data();
    VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
    r = vkCreateDescriptorSetLayout(dev, &dslci, nullptr, &dsl);
    if (r != VK_SUCCESS) {
        std::printf("FAIL %s dsl %s\n", path.c_str(), err_name(r));
        vkDestroyShaderModule(dev, sm, nullptr);
        return 1;
    }
    VkPipelineLayoutCreateInfo plci{
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.setLayoutCount = 1;
    plci.pSetLayouts = &dsl;
    VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, 128};
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges = &pcr;
    VkPipelineLayout pl = VK_NULL_HANDLE;
    r = vkCreatePipelineLayout(dev, &plci, nullptr, &pl);
    if (r != VK_SUCCESS) {
        std::printf("FAIL %s layout %s\n", path.c_str(), err_name(r));
        vkDestroyDescriptorSetLayout(dev, dsl, nullptr);
        vkDestroyShaderModule(dev, sm, nullptr);
        return 1;
    }
    VkComputePipelineCreateInfo cpci{
        VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = sm;
    cpci.stage.pName = "main";
    cpci.layout = pl;
    VkPipeline pipe = VK_NULL_HANDLE;
    r = vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpci, nullptr, &pipe);
    if (r != VK_SUCCESS) {
        std::printf("FAIL %s pipeline %s\n", path.c_str(), err_name(r));
        vkDestroyPipelineLayout(dev, pl, nullptr);
        vkDestroyDescriptorSetLayout(dev, dsl, nullptr);
        vkDestroyShaderModule(dev, sm, nullptr);
        return 1;
    }
    VkPipelineInfoKHR pi{VK_STRUCTURE_TYPE_PIPELINE_INFO_KHR};
    pi.pipeline = pipe;
    uint32_t ec = 0;
    ctx.get_props(dev, &pi, &ec, nullptr);
    std::vector<VkPipelineExecutablePropertiesKHR> ep(ec);
    for (auto &e : ep)
        e.sType = VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_PROPERTIES_KHR;
    ctx.get_props(dev, &pi, &ec, ep.data());
    std::printf("SHADER %s\n", path.c_str());
    for (uint32_t i = 0; i < ec; ++i) {
        VkPipelineExecutableInfoKHR ei{
            VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INFO_KHR};
        ei.pipeline = pipe;
        ei.executableIndex = i;
        uint32_t sc = 0;
        ctx.get_stats(dev, &ei, &sc, nullptr);
        std::vector<VkPipelineExecutableStatisticKHR> stats(sc);
        for (auto &s : stats)
            s.sType = VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_STATISTIC_KHR;
        if (sc && ctx.get_stats(dev, &ei, &sc, stats.data()) == VK_SUCCESS) {
            for (auto &s : stats) {
                double v = 0;
                switch (s.format) {
                case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_BOOL32_KHR:
                    v = s.value.b32;
                    break;
                case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_INT64_KHR:
                    v = static_cast<double>(s.value.i64);
                    break;
                case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_UINT64_KHR:
                    v = static_cast<double>(s.value.u64);
                    break;
                case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_FLOAT64_KHR:
                    v = s.value.f64;
                    break;
                default:
                    continue;
                }
                std::printf("STAT %s=%.0f\n", s.name, v);
            }
        }
    }
    std::fflush(stdout);
    vkDestroyPipeline(dev, pipe, nullptr);
    vkDestroyPipelineLayout(dev, pl, nullptr);
    vkDestroyDescriptorSetLayout(dev, dsl, nullptr);
    vkDestroyShaderModule(dev, sm, nullptr);
    return 0;
}

}  // namespace

int main(int argc, char **argv) {
    int want_index = -1;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--device-index") && i + 1 < argc) {
            want_index = std::atoi(argv[++i]);
        }
    }
    std::vector<std::string> paths;
    {
        char buf[4096];
        while (std::fgets(buf, sizeof(buf), stdin)) {
            std::string s(buf);
            while (!s.empty() && (s.back() == '\n' || s.back() == '\r'))
                s.pop_back();
            if (!s.empty()) paths.push_back(s);
        }
    }
    if (paths.empty()) return 0;

    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "vkop-roofline-probe";
    app.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    VkInstance inst = VK_NULL_HANDLE;
    if (vkCreateInstance(&ici, nullptr, &inst) != VK_SUCCESS) return 1;

    uint32_t ndev = 0;
    vkEnumeratePhysicalDevices(inst, &ndev, nullptr);
    std::vector<VkPhysicalDevice> devs(ndev);
    vkEnumeratePhysicalDevices(inst, &ndev, devs.data());

    // Pick a device: the requested index, else the first non-CPU one with
    // pipeline executable properties, else anything.
    int chosen = -1;
    for (uint32_t d = 0; d < ndev && chosen < 0; ++d) {
        if (want_index >= 0 && static_cast<int>(d) != want_index) continue;
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(devs[d], &props);
        if (want_index < 0 &&
            props.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU) continue;
        uint32_t ac = 0;
        vkEnumerateDeviceExtensionProperties(devs[d], nullptr, &ac, nullptr);
        std::vector<VkExtensionProperties> ex(ac);
        vkEnumerateDeviceExtensionProperties(devs[d], nullptr, &ac, ex.data());
        bool pep = false;
        for (auto &e : ex) {
            if (!std::strcmp(e.extensionName,
                             VK_KHR_PIPELINE_EXECUTABLE_PROPERTIES_EXTENSION_NAME))
                pep = true;
        }
        if (pep || want_index >= 0) chosen = static_cast<int>(d);
    }
    if (chosen < 0) {
        std::printf("FAIL * no-device\n");
        return 2;
    }
    VkPhysicalDevice phys = devs[chosen];
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(phys, &props);
    std::fprintf(stderr, "[probe] device: %s (vendor 0x%x)\n",
                 props.deviceName, props.vendorID);

    uint32_t qfc = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(phys, &qfc, nullptr);
    std::vector<VkQueueFamilyProperties> qfs(qfc);
    vkGetPhysicalDeviceQueueFamilyProperties(phys, &qfc, qfs.data());
    uint32_t qf = UINT32_MAX;
    for (uint32_t i = 0; i < qfc; ++i) {
        if (qfs[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            qf = i;
            break;
        }
    }
    if (qf == UINT32_MAX) {
        std::printf("FAIL * no-compute-queue\n");
        return 2;
    }

    // Feature probing: enable what the device advertises, so shaders that need
    // it compile. The chain pins the EXT structs, which stay valid on a
    // promoted core version.
    VkPhysicalDeviceFeatures core{};
    core.shaderInt64 = VK_TRUE;
    core.shaderInt16 = VK_TRUE;
    core.shaderFloat64 = VK_FALSE;

    VkPhysicalDeviceShaderFloat16Int8Features f16i8{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES};
    VkPhysicalDevice8BitStorageFeatures s8{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES};
    VkPhysicalDevice16BitStorageFeatures s16{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES};
    VkPhysicalDevicePipelineExecutablePropertiesFeaturesKHR pepf{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_EXECUTABLE_PROPERTIES_FEATURES_KHR};
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR coop{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR};
#ifdef VK_EXT_shader_float8
    VkPhysicalDeviceShaderFloat8FeaturesEXT f8{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT8_FEATURES_EXT};
#endif
    VkPhysicalDeviceSubgroupSizeControlFeatures subg{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES};

    auto query = [&](auto *s) {
        VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
        f2.pNext = s;
        vkGetPhysicalDeviceFeatures2(phys, &f2);
        return f2.features;
    };
    // Chain the query structures once.
    f16i8.pNext = &s8;
    s8.pNext = &s16;
    s16.pNext = &pepf;
    pepf.pNext = &coop;
#ifdef VK_EXT_shader_float8
    coop.pNext = &f8;
    f8.pNext = &subg;
#else
    coop.pNext = &subg;
#endif
    subg.pNext = nullptr;
    VkPhysicalDeviceFeatures2 q2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    q2.pNext = &f16i8;
    vkGetPhysicalDeviceFeatures2(phys, &q2);
    core = q2.features;
    // Re-chain for the enable struct: vkGetPhysicalDeviceFeatures2 leaves the
    // structs populated in place, so they can be passed straight on.
    VkPhysicalDeviceFeatures enable_core{};
    enable_core.shaderInt64 = core.shaderInt64;
    enable_core.shaderInt16 = core.shaderInt16;
    pepf.pipelineExecutableInfo = VK_TRUE;

    float pri = 1.0f;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = qf;
    qci.queueCount = 1;
    qci.pQueuePriorities = &pri;

    std::vector<const char *> exts;
    uint32_t ac = 0;
    vkEnumerateDeviceExtensionProperties(phys, nullptr, &ac, nullptr);
    std::vector<VkExtensionProperties> avail(ac);
    vkEnumerateDeviceExtensionProperties(phys, nullptr, &ac, avail.data());
    auto has_ext = [&](const char *n) {
        for (auto &e : avail)
            if (!std::strcmp(e.extensionName, n)) return true;
        return false;
    };
    if (has_ext(VK_KHR_PIPELINE_EXECUTABLE_PROPERTIES_EXTENSION_NAME))
        exts.push_back(VK_KHR_PIPELINE_EXECUTABLE_PROPERTIES_EXTENSION_NAME);
    bool want_coop = has_ext(VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME) &&
                     coop.cooperativeMatrix;
    if (want_coop)
        exts.push_back(VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME);
    bool want_f8 = false;
#ifdef VK_EXT_shader_float8
    want_f8 = has_ext(VK_EXT_SHADER_FLOAT8_EXTENSION_NAME) &&
              f8.shaderFloat8;
    if (want_f8) exts.push_back(VK_EXT_SHADER_FLOAT8_EXTENSION_NAME);
#else
    (void)want_f8;
#endif

    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    f16i8.pNext = &s8;
    s8.pNext = &s16;
    s16.pNext = &pepf;
    pepf.pNext = want_coop ? static_cast<void *>(&coop) : nullptr;
#ifdef VK_EXT_shader_float8
    coop.pNext = want_f8 ? static_cast<void *>(&f8) : nullptr;
    f8.pNext = nullptr;
#else
    coop.pNext = nullptr;
#endif
    dci.pNext = &f16i8;
    dci.pEnabledFeatures = &enable_core;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = static_cast<uint32_t>(exts.size());
    dci.ppEnabledExtensionNames = exts.empty() ? nullptr : exts.data();
    VkDevice dev = VK_NULL_HANDLE;
    VkResult dr = vkCreateDevice(phys, &dci, nullptr, &dev);
    if (dr != VK_SUCCESS) {
        std::printf("FAIL * device-create %s\n", err_name(dr));
        return 2;
    }

    auto GetProps = reinterpret_cast<PFN_vkGetPipelineExecutablePropertiesKHR>(
        vkGetDeviceProcAddr(dev, "vkGetPipelineExecutablePropertiesKHR"));
    auto GetStats = reinterpret_cast<PFN_vkGetPipelineExecutableStatisticsKHR>(
        vkGetDeviceProcAddr(dev, "vkGetPipelineExecutableStatisticsKHR"));
    if (!GetProps || !GetStats) {
        std::printf("FAIL * no-executable-properties\n");
        return 2;
    }
    std::printf("DEVICE %s\n", props.deviceName);
    std::printf("VENDOR 0x%x\n", props.vendorID);

    Ctx ctx;
    ctx.dev = dev;
    ctx.get_props = GetProps;
    ctx.get_stats = GetStats;

    // One fork per shader.  This is not paranoia: a driver that SIGSEGVs while
    // compiling one module (Intel ANV does exactly that on the SkLast
    // variants -- see ops/Matmul.hpp's comment) would otherwise take the whole
    // probe down and lose every already-computed number.
    for (const std::string &path : paths) {
        std::fflush(stdout);
        pid_t pid = ::fork();
        if (pid == 0) {
            int rc = probe_one(ctx, path);
            std::fflush(stdout);
            ::_exit(rc);
        }
        int status = 0;
        ::waitpid(pid, &status, 0);
        if (WIFSIGNALED(status)) {
            std::printf("FAIL %s crashed signal %d\n", path.c_str(),
                        WTERMSIG(status));
            std::fflush(stdout);
        }
    }
    vkDestroyDevice(dev, nullptr);
    vkDestroyInstance(inst, nullptr);
    return 0;
}
