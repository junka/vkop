// VK_EXT_shader_float8 capability + correctness probe.
//
// Deliberately standalone: it only needs the Vulkan headers and loader from a
// SDK >= 1.4.317, not libvkop. That makes it usable as a first-contact test on
// a machine where the vkop tree has not been rebuilt against those headers yet.
//
// Two checks, because a device may pass one and fail the other:
//   1. storage: OpTypeFloat 8 Float8E4M3EXT, float -> E4M3 -> float round trip.
//   2. coopmat: Float8CooperativeMatrixEXT, one 16x16x32 E4M3 x E4M3 -> FP32
//      coopMatMulAdd.
//
// Usage: f8_probe <f8_storage.spv> <f8_coopmat.spv>
// Exit: 0 = both pass or cleanly skipped, 1 = a check failed, 2 = usage/load.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <vulkan/vulkan.h>

#ifndef VK_EXT_shader_float8
#error "VK_EXT_shader_float8 is not in these headers: use an SDK >= 1.4.317"
#endif

namespace {

int g_failures = 0;

void check(VkResult r, const char *what, int line) {
    if (r != VK_SUCCESS) {
        std::fprintf(stderr, "FAIL %s -> VkResult %d (line %d)\n", what,
                     static_cast<int>(r), line);
        std::exit(2);
    }
}
#define VK_CHECK(x) check((x), #x, __LINE__)

// ---- E4M3 (OCP, bias 7, no infinities, 0x7f is NaN) ----------------------
// Values used below are all exactly representable, so this never rounds.
unsigned char f32_to_e4m3_exact(float f) {
    if (f == 0.0f) {
        return std::signbit(f) ? 0x80 : 0x00;
    }
    unsigned bits;
    std::memcpy(&bits, &f, sizeof(bits));
    int sign = (bits >> 31) & 1;
    int exp = static_cast<int>((bits >> 23) & 0xff) - 127 + 7;
    unsigned mant = bits & 0x7fffffu;
    // 3 mantissa bits: the low 20 must be zero for an exact representation.
    if ((mant & 0xfffffu) != 0) {
        std::fprintf(stderr, "probe bug: %g is not exactly E4M3\n", f);
        std::exit(2);
    }
    if (exp < 1 || exp > 15) {
        std::fprintf(stderr, "probe bug: %g out of E4M3 range\n", f);
        std::exit(2);
    }
    return static_cast<unsigned char>((sign << 7) | (exp << 3) |
                                      ((mant >> 20) & 7u));
}

float e4m3_to_f32(unsigned char b) {
    int sign = (b >> 7) & 1;
    int exp = (b >> 3) & 0xf;
    int mant = b & 7;
    if (exp == 0) {
        return sign ? -0.0f : 0.0f;
    }
    float v = std::ldexp(1.0f + mant / 8.0f, exp - 7);
    return sign ? -v : v;
}

struct Device {
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice phys = VK_NULL_HANDLE;
    VkDevice dev = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    uint32_t queue_family = 0;
    VkPhysicalDeviceMemoryProperties mem_props{};
    uint32_t instance_api = 0;
    uint32_t device_api = 0;
    bool has_float8 = false;
    bool has_coopmat = false;
    bool float8_feature = false;
    bool float8_coopmat_feature = false;
    bool coopmat_feature = false;
    std::vector<VkCooperativeMatrixPropertiesKHR> coopmat_props;
};

Device open_device() {
    Device d;

    uint32_t loader_api = VK_API_VERSION_1_0;
    VK_CHECK(vkEnumerateInstanceVersion(&loader_api));
    const uint32_t want_1_4 = VK_MAKE_API_VERSION(0, 1, 4, 0);
    const uint32_t use = loader_api < want_1_4 ? loader_api : want_1_4;
    std::printf("[f8probe] loader %u.%u.%u, creating instance at %u.%u.%u\n",
                VK_API_VERSION_MAJOR(loader_api),
                VK_API_VERSION_MINOR(loader_api),
                VK_API_VERSION_PATCH(loader_api), VK_API_VERSION_MAJOR(use),
                VK_API_VERSION_MINOR(use), VK_API_VERSION_PATCH(use));

    VkApplicationInfo app = {};
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName = "vkop_float8_probe";
    app.apiVersion = use;
    VkInstanceCreateInfo ici = {};
    ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ici.pApplicationInfo = &app;
    VK_CHECK(vkCreateInstance(&ici, nullptr, &d.instance));
    d.instance_api = use;

    uint32_t count = 0;
    VK_CHECK(vkEnumeratePhysicalDevices(d.instance, &count, nullptr));
    if (count == 0) {
        std::printf("[f8probe] SKIP: no physical device\n");
        std::exit(0);
    }
    std::vector<VkPhysicalDevice> phys(count);
    VK_CHECK(vkEnumeratePhysicalDevices(d.instance, &count, phys.data()));

    // Prefer a discrete GPU; fall back to the first device that has the
    // extension, then to device 0 so the report still names what was found.
    VkPhysicalDevice chosen = phys[0];
    for (VkPhysicalDevice p : phys) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(p, &props);
        uint32_t next = 0;
        vkEnumerateDeviceExtensionProperties(p, nullptr, &next, nullptr);
        std::vector<VkExtensionProperties> exts(next);
        vkEnumerateDeviceExtensionProperties(p, nullptr, &next, exts.data());
        bool f8 = false;
        for (const auto &e : exts) {
            if (!std::strcmp(e.extensionName,
                             VK_EXT_SHADER_FLOAT8_EXTENSION_NAME)) {
                f8 = true;
            }
        }
        std::printf("[f8probe] device: %s (api %u.%u.%u, type %d, float8=%d)\n",
                    props.deviceName, VK_API_VERSION_MAJOR(props.apiVersion),
                    VK_API_VERSION_MINOR(props.apiVersion),
                    VK_API_VERSION_PATCH(props.apiVersion),
                    static_cast<int>(props.deviceType), f8 ? 1 : 0);
        if (f8 && props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
            chosen = p;
            break;
        }
        if (f8) {
            chosen = p;
        }
    }
    d.phys = chosen;
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(d.phys, &props);
    d.device_api = props.apiVersion;
    std::printf("[f8probe] using %s\n", props.deviceName);

    uint32_t next = 0;
    VK_CHECK(vkEnumerateDeviceExtensionProperties(d.phys, nullptr, &next,
                                                  nullptr));
    std::vector<VkExtensionProperties> exts(next);
    VK_CHECK(vkEnumerateDeviceExtensionProperties(d.phys, nullptr, &next,
                                                  exts.data()));
    for (const auto &e : exts) {
        if (!std::strcmp(e.extensionName,
                         VK_EXT_SHADER_FLOAT8_EXTENSION_NAME)) {
            d.has_float8 = true;
        }
        if (!std::strcmp(e.extensionName,
                         VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME)) {
            d.has_coopmat = true;
        }
    }
    std::printf("[f8probe] extensions: VK_EXT_shader_float8=%d "
                "VK_KHR_cooperative_matrix=%d\n",
                d.has_float8 ? 1 : 0, d.has_coopmat ? 1 : 0);
    if (!d.has_float8) {
        std::printf("[f8probe] SKIP: VK_EXT_shader_float8 not advertised\n");
        std::exit(0);
    }

    VkPhysicalDeviceShaderFloat8FeaturesEXT f8 = {};
    f8.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT8_FEATURES_EXT;
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR cm = {};
    cm.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR;
    VkPhysicalDeviceFeatures2 q = {};
    q.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    if (d.has_coopmat) {
        q.pNext = &f8;
        f8.pNext = &cm;
    } else {
        q.pNext = &f8;
    }
    vkGetPhysicalDeviceFeatures2(d.phys, &q);
    d.float8_feature = f8.shaderFloat8 == VK_TRUE;
    d.float8_coopmat_feature = f8.shaderFloat8CooperativeMatrix == VK_TRUE;
    d.coopmat_feature = cm.cooperativeMatrix == VK_TRUE;
    std::printf("[f8probe] features: shaderFloat8=%d "
                "shaderFloat8CooperativeMatrix=%d cooperativeMatrix=%d\n",
                d.float8_feature ? 1 : 0, d.float8_coopmat_feature ? 1 : 0,
                d.coopmat_feature ? 1 : 0);
    if (!d.float8_feature) {
        std::printf("[f8probe] SKIP: shaderFloat8 feature is false\n");
        std::exit(0);
    }

    if (d.has_coopmat && d.coopmat_feature) {
        VkPhysicalDeviceCooperativeMatrixFeaturesKHR cf = {};
        cf.sType =
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR;
        VkPhysicalDeviceProperties2 p2 = {};
        p2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
        VkPhysicalDeviceCooperativeMatrixPropertiesKHR cp = {};
        cp.sType =
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_PROPERTIES_KHR;
        p2.pNext = &cp;
        vkGetPhysicalDeviceProperties2(d.phys, &p2);
        std::printf("[f8probe] coopmat supported stages = 0x%x\n",
                    cp.cooperativeMatrixSupportedStages);

        auto get_props = reinterpret_cast<
            PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR>(
            vkGetInstanceProcAddr(d.instance,
                                  "vkGetPhysicalDeviceCooperativeMatrixPropert"
                                  "iesKHR"));
        if (get_props) {
            uint32_t n = 0;
            if (get_props(d.phys, &n, nullptr) == VK_SUCCESS && n > 0) {
                d.coopmat_props.resize(n);
                for (auto &p : d.coopmat_props) {
                    p.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_PROPERTIES_KHR;
                }
                get_props(d.phys, &n, d.coopmat_props.data());
            }
        }
        bool listed = false;
        for (const auto &p : d.coopmat_props) {
            if (p.AType == VK_COMPONENT_TYPE_FLOAT8_E4M3_EXT &&
                p.BType == VK_COMPONENT_TYPE_FLOAT8_E4M3_EXT &&
                p.MSize == 16 && p.NSize == 16 && p.KSize == 32 &&
                p.scope == VK_SCOPE_SUBGROUP_KHR) {
                listed = true;
            }
            if (p.AType == VK_COMPONENT_TYPE_FLOAT8_E4M3_EXT ||
                p.BType == VK_COMPONENT_TYPE_FLOAT8_E4M3_EXT) {
                std::printf("[f8probe] coopmat fp8 config: %ux%ux%u A=%d B=%d "
                            "C=%d R=%d scope=%d\n",
                            p.MSize, p.NSize, p.KSize,
                            static_cast<int>(p.AType),
                            static_cast<int>(p.BType),
                            static_cast<int>(p.CType),
                            static_cast<int>(p.ResultType),
                            static_cast<int>(p.scope));
            }
        }
        std::printf("[f8probe] 16x16x32 E4M3xE4M3 listed = %d\n",
                    listed ? 1 : 0);
    }

    uint32_t nq = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(d.phys, &nq, nullptr);
    std::vector<VkQueueFamilyProperties> qfp(nq);
    vkGetPhysicalDeviceQueueFamilyProperties(d.phys, &nq, qfp.data());
    for (uint32_t i = 0; i < nq; i++) {
        if (qfp[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            d.queue_family = i;
            break;
        }
    }
    vkGetPhysicalDeviceMemoryProperties(d.phys, &d.mem_props);

    std::vector<const char *> enabled;
    enabled.push_back(VK_EXT_SHADER_FLOAT8_EXTENSION_NAME);
    VkPhysicalDeviceShaderFloat8FeaturesEXT ef8 = {};
    ef8.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT8_FEATURES_EXT;
    ef8.shaderFloat8 = VK_TRUE;
    ef8.shaderFloat8CooperativeMatrix =
        d.float8_coopmat_feature ? VK_TRUE : VK_FALSE;

    VkPhysicalDeviceCooperativeMatrixFeaturesKHR ecm = {};
    ecm.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR;
    VkPhysicalDeviceVulkanMemoryModelFeatures vmm = {};
    vmm.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_MEMORY_MODEL_FEATURES;
    vmm.vulkanMemoryModel = VK_TRUE;
    vmm.vulkanMemoryModelDeviceScope = VK_TRUE;

    // Enable coopmat only when the kernel will actually be dispatched, so a
    // device without it does not fail vkCreateDevice on the storage test.
    if (d.has_coopmat && d.coopmat_feature && d.float8_coopmat_feature &&
        !d.coopmat_props.empty()) {
        enabled.push_back(VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME);
        ecm.cooperativeMatrix = VK_TRUE;
        ef8.pNext = &ecm;
        ecm.pNext = &vmm;
    }

    float priority = 1.0f;
    VkDeviceQueueCreateInfo qci = {};
    qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qci.queueFamilyIndex = d.queue_family;
    qci.queueCount = 1;
    qci.pQueuePriorities = &priority;
    VkDeviceCreateInfo dci = {};
    dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dci.pNext = &ef8;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = static_cast<uint32_t>(enabled.size());
    dci.ppEnabledExtensionNames = enabled.data();
    VkResult r = vkCreateDevice(d.phys, &dci, nullptr, &d.dev);
    if (r != VK_SUCCESS) {
        std::fprintf(stderr, "FAIL vkCreateDevice -> %d\n",
                     static_cast<int>(r));
        std::exit(1);
    }
    vkGetDeviceQueue(d.dev, d.queue_family, 0, &d.queue);
    std::printf("[f8probe] device created\n");
    return d;
}

struct Buffer {
    VkBuffer buf = VK_NULL_HANDLE;
    VkDeviceMemory mem = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
    void *mapped = nullptr;
};

Buffer make_buffer(Device &d, VkDeviceSize size) {
    Buffer b;
    b.size = size;
    VkBufferCreateInfo bci = {};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size = size;
    bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    VK_CHECK(vkCreateBuffer(d.dev, &bci, nullptr, &b.buf));
    VkMemoryRequirements mr;
    vkGetBufferMemoryRequirements(d.dev, b.buf, &mr);
    uint32_t type = UINT32_MAX;
    for (uint32_t t = 0; t < d.mem_props.memoryTypeCount; t++) {
        const VkMemoryPropertyFlags want = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                           VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        if ((mr.memoryTypeBits & (1u << t)) &&
            (d.mem_props.memoryTypes[t].propertyFlags & want) == want) {
            type = t;
            break;
        }
    }
    VkMemoryAllocateInfo mai = {};
    mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.allocationSize = mr.size;
    mai.memoryTypeIndex = type;
    VK_CHECK(vkAllocateMemory(d.dev, &mai, nullptr, &b.mem));
    VK_CHECK(vkBindBufferMemory(d.dev, b.buf, b.mem, 0));
    VK_CHECK(vkMapMemory(d.dev, b.mem, 0, size, 0, &b.mapped));
    std::memset(b.mapped, 0, size);
    return b;
}

VkShaderModule load_module(Device &d, const char *path) {
    FILE *fp = std::fopen(path, "rb");
    if (!fp) {
        std::fprintf(stderr, "cannot open shader %s\n", path);
        std::exit(2);
    }
    std::fseek(fp, 0, SEEK_END);
    long len = std::ftell(fp);
    std::fseek(fp, 0, SEEK_SET);
    std::vector<char> code(static_cast<size_t>(len));
    if (std::fread(code.data(), 1, code.size(), fp) != code.size()) {
        std::fprintf(stderr, "short read on %s\n", path);
        std::exit(2);
    }
    std::fclose(fp);
    VkShaderModuleCreateInfo ci = {};
    ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    ci.codeSize = code.size();
    ci.pCode = reinterpret_cast<const uint32_t *>(code.data());
    VkShaderModule m = VK_NULL_HANDLE;
    VK_CHECK(vkCreateShaderModule(d.dev, &ci, nullptr, &m));
    return m;
}

struct Result {
    bool ran = false;
    bool pass = false;
    int bad = 0;
    int total = 0;
    float max_err = 0.0f;
};

// ---------------------------------------------------------------------------
// 1. storage-buffer E4M3 round trip
// ---------------------------------------------------------------------------
Result run_storage(Device &d, const char *spv, bool *pipeline_created) {
    Result res;
    const uint32_t n = 256;
    std::vector<float> in(n);
    for (uint32_t i = 0; i < n; i++) {
        // Exactly representable E4M3 values in [0.5, 8): sign/exponent/mantissa
        // coverage without any rounding on the way in.
        int m = static_cast<int>(i % 16);
        in[i] = static_cast<float>(1 + m / 8) *
                std::ldexp(1.0f, static_cast<int>(i % 6) - 1);
    }
    std::vector<float> want(n);
    for (uint32_t i = 0; i < n; i++) {
        want[i] = e4m3_to_f32(f32_to_e4m3_exact(in[i]));
    }

    Buffer bin = make_buffer(d, n * sizeof(float));
    Buffer bq = make_buffer(d, n);
    Buffer bout = make_buffer(d, n * sizeof(float));
    std::memcpy(bin.mapped, in.data(), n * sizeof(float));

    VkDescriptorSetLayoutBinding binds[3] = {};
    for (int i = 0; i < 3; i++) {
        binds[i].binding = static_cast<uint32_t>(i);
        binds[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        binds[i].descriptorCount = 1;
        binds[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dslci = {};
    dslci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dslci.bindingCount = 3;
    dslci.pBindings = binds;
    VkDescriptorSetLayout dsl;
    VK_CHECK(vkCreateDescriptorSetLayout(d.dev, &dslci, nullptr, &dsl));

    VkPushConstantRange pcr = {};
    pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcr.offset = 0;
    pcr.size = sizeof(uint32_t);
    VkPipelineLayoutCreateInfo plci = {};
    plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.setLayoutCount = 1;
    plci.pSetLayouts = &dsl;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges = &pcr;
    VkPipelineLayout pl;
    VK_CHECK(vkCreatePipelineLayout(d.dev, &plci, nullptr, &pl));

    VkShaderModule sm = load_module(d, spv);
    VkComputePipelineCreateInfo cpci = {};
    cpci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = sm;
    cpci.stage.pName = "main";
    cpci.layout = pl;
    VkPipeline pipe = VK_NULL_HANDLE;
    VkResult r =
        vkCreateComputePipelines(d.dev, VK_NULL_HANDLE, 1, &cpci, nullptr, &pipe);
    if (r != VK_SUCCESS) {
        std::printf("[f8probe] storage: pipeline REJECTED (VkResult %d)\n",
                    static_cast<int>(r));
        *pipeline_created = false;
        return res;
    }
    *pipeline_created = true;
    std::printf("[f8probe] storage: pipeline created\n");

    VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3};
    VkDescriptorPoolCreateInfo dpci = {};
    dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpci.maxSets = 1;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes = &ps;
    VkDescriptorPool dp;
    VK_CHECK(vkCreateDescriptorPool(d.dev, &dpci, nullptr, &dp));
    VkDescriptorSetAllocateInfo dsai = {};
    dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsai.descriptorPool = dp;
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts = &dsl;
    VkDescriptorSet ds;
    VK_CHECK(vkAllocateDescriptorSets(d.dev, &dsai, &ds));
    Buffer *bufs[3] = {&bin, &bq, &bout};
    VkDescriptorBufferInfo dbi[3];
    VkWriteDescriptorSet w[3];
    for (int i = 0; i < 3; i++) {
        dbi[i] = {};
        dbi[i].buffer = bufs[i]->buf;
        dbi[i].range = bufs[i]->size;
        w[i] = {};
        w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[i].dstSet = ds;
        w[i].dstBinding = static_cast<uint32_t>(i);
        w[i].descriptorCount = 1;
        w[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w[i].pBufferInfo = &dbi[i];
    }
    vkUpdateDescriptorSets(d.dev, 3, w, 0, nullptr);

    VkCommandPoolCreateInfo cpi = {};
    cpi.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cpi.queueFamilyIndex = d.queue_family;
    VkCommandPool cp;
    VK_CHECK(vkCreateCommandPool(d.dev, &cpi, nullptr, &cp));
    VkCommandBufferAllocateInfo cbai = {};
    cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cbai.commandPool = cp;
    cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = 1;
    VkCommandBuffer cb;
    VK_CHECK(vkAllocateCommandBuffers(d.dev, &cbai, &cb));
    VkCommandBufferBeginInfo cbbi = {};
    cbbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    VK_CHECK(vkBeginCommandBuffer(cb, &cbbi));
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &ds,
                            0, nullptr);
    vkCmdPushConstants(cb, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       sizeof(uint32_t), &n);
    vkCmdDispatch(cb, (n + 63) / 64, 1, 1);
    VK_CHECK(vkEndCommandBuffer(cb));
    VkSubmitInfo si = {};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cb;
    VK_CHECK(vkQueueSubmit(d.queue, 1, &si, VK_NULL_HANDLE));
    VK_CHECK(vkQueueWaitIdle(d.queue));

    res.ran = true;
    res.total = static_cast<int>(n);
    const float *out = static_cast<const float *>(bout.mapped);
    for (uint32_t i = 0; i < n; i++) {
        float e = std::fabs(out[i] - want[i]);
        if (e > res.max_err) {
            res.max_err = e;
        }
        if (e != 0.0f) {
            if (res.bad < 5) {
                std::printf("[f8probe] storage mismatch[%u] got %g want %g\n", i,
                            out[i], want[i]);
            }
            res.bad++;
        }
    }
    res.pass = res.bad == 0;
    return res;
}

// ---------------------------------------------------------------------------
// 2. E4M3 x E4M3 -> FP32 cooperative matrix
// ---------------------------------------------------------------------------
Result run_coopmat(Device &d, const char *spv, bool *pipeline_created) {
    Result res;
    const int M = 16, K = 32, N = 16;
    std::vector<unsigned char> a(M * K), b(K * N);
    std::vector<float> af(M * K), bf(K * N);
    for (int i = 0; i < M; i++) {
        for (int k = 0; k < K; k++) {
            // ((i + k) % 7) + 1: all exact, nonzero, and varied enough that a
            // transposed or zeroed result cannot match.
            float v = static_cast<float>(((i * 3 + k * 5) % 7) + 1);
            a[i * K + k] = f32_to_e4m3_exact(v);
            af[i * K + k] = e4m3_to_f32(a[i * K + k]);
        }
    }
    for (int k = 0; k < K; k++) {
        for (int j = 0; j < N; j++) {
            float v = static_cast<float>(((k * 2 + j * 3) % 7) + 1);
            b[k * N + j] = f32_to_e4m3_exact(v);
            bf[k * N + j] = e4m3_to_f32(b[k * N + j]);
        }
    }
    std::vector<float> want(M * N, 0.0f);
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float s = 0.0f;
            for (int k = 0; k < K; k++) {
                s += af[i * K + k] * bf[k * N + j];
            }
            want[i * N + j] = s;
        }
    }

    Buffer ba = make_buffer(d, a.size());
    Buffer bb = make_buffer(d, b.size());
    Buffer bc = make_buffer(d, M * N * sizeof(float));
    std::memcpy(ba.mapped, a.data(), a.size());
    std::memcpy(bb.mapped, b.data(), b.size());

    VkDescriptorSetLayoutBinding binds[3] = {};
    for (int i = 0; i < 3; i++) {
        binds[i].binding = static_cast<uint32_t>(i);
        binds[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        binds[i].descriptorCount = 1;
        binds[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dslci = {};
    dslci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dslci.bindingCount = 3;
    dslci.pBindings = binds;
    VkDescriptorSetLayout dsl;
    VK_CHECK(vkCreateDescriptorSetLayout(d.dev, &dslci, nullptr, &dsl));
    VkPipelineLayoutCreateInfo plci = {};
    plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.setLayoutCount = 1;
    plci.pSetLayouts = &dsl;
    VkPipelineLayout pl;
    VK_CHECK(vkCreatePipelineLayout(d.dev, &plci, nullptr, &pl));

    VkShaderModule sm = load_module(d, spv);
    VkComputePipelineCreateInfo cpci = {};
    cpci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = sm;
    cpci.stage.pName = "main";
    cpci.layout = pl;
    VkPipeline pipe = VK_NULL_HANDLE;
    VkResult r =
        vkCreateComputePipelines(d.dev, VK_NULL_HANDLE, 1, &cpci, nullptr, &pipe);
    if (r != VK_SUCCESS) {
        std::printf("[f8probe] coopmat: pipeline REJECTED (VkResult %d)\n",
                    static_cast<int>(r));
        *pipeline_created = false;
        return res;
    }
    *pipeline_created = true;
    std::printf("[f8probe] coopmat: pipeline created\n");

    VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3};
    VkDescriptorPoolCreateInfo dpci = {};
    dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpci.maxSets = 1;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes = &ps;
    VkDescriptorPool dp;
    VK_CHECK(vkCreateDescriptorPool(d.dev, &dpci, nullptr, &dp));
    VkDescriptorSetAllocateInfo dsai = {};
    dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsai.descriptorPool = dp;
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts = &dsl;
    VkDescriptorSet ds;
    VK_CHECK(vkAllocateDescriptorSets(d.dev, &dsai, &ds));
    Buffer *bufs[3] = {&ba, &bb, &bc};
    VkDescriptorBufferInfo dbi[3];
    VkWriteDescriptorSet w[3];
    for (int i = 0; i < 3; i++) {
        dbi[i] = {};
        dbi[i].buffer = bufs[i]->buf;
        dbi[i].range = bufs[i]->size;
        w[i] = {};
        w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[i].dstSet = ds;
        w[i].dstBinding = static_cast<uint32_t>(i);
        w[i].descriptorCount = 1;
        w[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w[i].pBufferInfo = &dbi[i];
    }
    vkUpdateDescriptorSets(d.dev, 3, w, 0, nullptr);

    VkCommandPoolCreateInfo cpi = {};
    cpi.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cpi.queueFamilyIndex = d.queue_family;
    VkCommandPool cp;
    VK_CHECK(vkCreateCommandPool(d.dev, &cpi, nullptr, &cp));
    VkCommandBufferAllocateInfo cbai = {};
    cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cbai.commandPool = cp;
    cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = 1;
    VkCommandBuffer cb;
    VK_CHECK(vkAllocateCommandBuffers(d.dev, &cbai, &cb));
    VkCommandBufferBeginInfo cbbi = {};
    cbbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    VK_CHECK(vkBeginCommandBuffer(cb, &cbbi));
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &ds,
                            0, nullptr);
    vkCmdDispatch(cb, 1, 1, 1);
    VK_CHECK(vkEndCommandBuffer(cb));
    VkSubmitInfo si = {};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cb;
    VK_CHECK(vkQueueSubmit(d.queue, 1, &si, VK_NULL_HANDLE));
    VK_CHECK(vkQueueWaitIdle(d.queue));

    res.ran = true;
    res.total = M * N;
    std::vector<float> out(M * N);
    std::memcpy(out.data(), bc.mapped, M * N * sizeof(float));
    for (int i = 0; i < M * N; i++) {
        float e = std::fabs(out[i] - want[i]);
        if (e > res.max_err) {
            res.max_err = e;
        }
        // The accumulate path is exact for these magnitudes (all sums are
        // small integers); a nonzero error means wrong data was read.
        if (e > 1e-3f) {
            if (res.bad < 5) {
                std::printf("[f8probe] coopmat mismatch[%d] got %g want %g\n", i,
                            out[i], want[i]);
            }
            res.bad++;
        }
    }
    res.pass = res.bad == 0;
    if (res.pass) {
        std::printf("[f8probe] coopmat sample: got[0]=%g want[0]=%g "
                    "got[5]=%g want[5]=%g\n",
                    out[0], want[0], out[5], want[5]);
    }
    return res;
}

} // namespace

int main(int argc, char **argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: %s <f8_storage.spv> <f8_coopmat.spv>\n",
                     argv[0]);
        return 2;
    }
    Device d = open_device();

    bool storage_pipe = false;
    Result storage = run_storage(d, argv[1], &storage_pipe);
    if (storage.ran) {
        std::printf("[f8probe] storage RESULT %s: %d/%d bad, maxerr=%g\n",
                    storage.pass ? "PASS" : "FAIL", storage.bad,
                    storage.total, storage.max_err);
    }
    if (!storage_pipe) {
        g_failures++;
    }

    if (d.has_coopmat && d.coopmat_feature && d.float8_coopmat_feature) {
        bool coop_pipe = false;
        Result coop = run_coopmat(d, argv[2], &coop_pipe);
        if (coop.ran) {
            std::printf("[f8probe] coopmat RESULT %s: %d/%d bad, maxerr=%g\n",
                        coop.pass ? "PASS" : "FAIL", coop.bad, coop.total,
                        coop.max_err);
        }
        if (!coop_pipe) {
            g_failures++;
        }
    } else {
        std::printf("[f8probe] coopmat SKIP: extension or feature absent "
                    "(storage test only)\n");
    }

    std::printf("[f8probe] OVERALL %s\n", g_failures ? "FAIL" : "PASS");
    std::fflush(stdout);
    return g_failures ? 1 : 0;
}
