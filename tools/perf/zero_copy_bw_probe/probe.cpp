// Zero-copy bandwidth probe: staging vs host-import vs managed vs VRAM baseline.
// See dual_gpu_zero_copy.md. Build instructions in README.md next to this file.
#include <vulkan/vulkan.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CHECK(x)                                                              \
    do {                                                                      \
        VkResult r_ = (x);                                                    \
        if (r_ != VK_SUCCESS) {                                               \
            fprintf(stderr, "FAIL %s = %d at line %d\n", #x, r_, __LINE__);   \
            exit(1);                                                          \
        }                                                                     \
    } while (0)

#include "read_sum.spv.h"

static uint64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

static const uint32_t THREADS_FWD = 256;
static const uint32_t GROUPS_FWD = 1024;

struct Mem {
    VkDeviceMemory mem = VK_NULL_HANDLE;
    void* mapped = nullptr;
};

static uint32_t find_type(VkPhysicalDevice pd, uint32_t bits, VkFlags req,
                          int dl_pref) { // 1=require DL, 0=require non-DL, -1=prefer non-DL
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(pd, &mp);
    uint32_t fallback = UINT32_MAX;
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++) {
        if (!(bits & (1u << i))) continue;
        if ((mp.memoryTypes[i].propertyFlags & req) != req) continue;
        bool dl = mp.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
        if (dl_pref < 0) {
            if (!dl) return i;
            if (fallback == UINT32_MAX) fallback = i;
        } else if (dl == (dl_pref == 1)) {
            return i;
        }
    }
    if (fallback != UINT32_MAX) return fallback;
    fprintf(stderr, "no memory type for req=0x%x dl_pref=%d\n", req, dl_pref);
    exit(1);
}

static Mem alloc_mem(VkDevice dev, VkPhysicalDevice pd, VkDeviceSize size,
                     uint32_t type, void* host_ptr = nullptr) {
    Mem m;
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = size;
    ai.memoryTypeIndex = type;
    VkImportMemoryHostPointerInfoEXT imp{VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT};
    if (host_ptr) {
        imp.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;
        imp.pHostPointer = host_ptr;
        ai.pNext = &imp;
    }
    CHECK(vkAllocateMemory(dev, &ai, nullptr, &m.mem));
    return m;
}

struct Buf {
    VkBuffer buf = VK_NULL_HANDLE;
    Mem mem;
};

static Buf make_buf(VkDevice dev, VkPhysicalDevice pd, VkDeviceSize size,
                    uint32_t type, bool map, void* host_ptr = nullptr) {
    Buf b;
    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.size = size;
    bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
               VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    CHECK(vkCreateBuffer(dev, &bi, nullptr, &b.buf));
    VkMemoryRequirements mr;
    vkGetBufferMemoryRequirements(dev, b.buf, &mr);
    b.mem = alloc_mem(dev, pd, std::max(mr.size, size), type, host_ptr);
    CHECK(vkBindBufferMemory(dev, b.buf, b.mem.mem, 0));
    if (map) CHECK(vkMapMemory(dev, b.mem.mem, 0, VK_WHOLE_SIZE, 0, &b.mem.mapped));
    return b;
}

static void fill_pattern(void* p, size_t bytes) {
    uint8_t* b = static_cast<uint8_t*>(p);
    for (size_t i = 0; i < bytes; i++) b[i] = uint8_t((i * 31 + 7) & 0xff);
}

static uint64_t expected_sum(const void* p, size_t bytes) {
    // mirror the kernel: per-thread uint32 accumulation (mod 2^32), grid-stride
    const uint8_t* b = static_cast<const uint8_t*>(p);
    size_t n = bytes / 16;
    const uint32_t total_threads = GROUPS_FWD * THREADS_FWD;
    uint64_t s = 0;
    for (uint32_t t = 0; t < total_threads; t++) {
        uint32_t acc = 0;
        for (size_t i = t; i < n; i += total_threads) {
            uint32_t w[4];
            memcpy(w, b + i * 16, 16);
            acc += w[0] + w[1] + w[2] + w[3];
        }
        s += acc;
    }
    return s;
}

struct Device {
    VkPhysicalDevice pd;
    VkDevice dev;
    VkQueue queue;
    uint32_t qfam;
    VkCommandPool pool;
    VkCommandBuffer cb;
    VkFence fence;
    VkQueryPool qpool;
    float ts_period;
    VkPhysicalDeviceMemoryProperties mp;
    std::string name;
};

struct Pipeline {
    VkDescriptorSetLayout dsl;
    VkPipelineLayout pl;
    VkPipeline pipe;
    VkDescriptorPool dpool;
};

static Pipeline make_pipeline(VkDevice dev) {
    Pipeline p;
    VkShaderModuleCreateInfo smi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smi.codeSize = sizeof(read_sum_spv);
    smi.pCode = reinterpret_cast<const uint32_t*>(read_sum_spv);
    VkShaderModule sm;
    CHECK(vkCreateShaderModule(dev, &smi, nullptr, &sm));

    VkDescriptorSetLayoutBinding binds[2] = {};
    for (int i = 0; i < 2; i++) {
        binds[i].binding = i;
        binds[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        binds[i].descriptorCount = 1;
        binds[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dsli{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dsli.bindingCount = 2;
    dsli.pBindings = binds;
    CHECK(vkCreateDescriptorSetLayout(dev, &dsli, nullptr, &p.dsl));

    VkPushConstantRange pc{VK_SHADER_STAGE_COMPUTE_BIT, 0, 4};
    VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pli.setLayoutCount = 1;
    pli.pSetLayouts = &p.dsl;
    pli.pushConstantRangeCount = 1;
    pli.pPushConstantRanges = &pc;
    CHECK(vkCreatePipelineLayout(dev, &pli, nullptr, &p.pl));

    VkComputePipelineCreateInfo cpi{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpi.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpi.stage.module = sm;
    cpi.stage.pName = "main";
    cpi.layout = p.pl;
    CHECK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpi, nullptr, &p.pipe));
    vkDestroyShaderModule(dev, sm, nullptr);

    VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 16};
    VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpi.maxSets = 8;
    dpi.poolSizeCount = 1;
    dpi.pPoolSizes = &ps;
    CHECK(vkCreateDescriptorPool(dev, &dpi, nullptr, &p.dpool));
    return p;
}

static VkDescriptorSet bind_set(VkDevice dev, Pipeline& p, VkBuffer src,
                                VkDeviceSize src_size, VkBuffer dst,
                                VkDeviceSize dst_size) {
    VkDescriptorSet set;
    VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dai.descriptorPool = p.dpool;
    dai.descriptorSetCount = 1;
    dai.pSetLayouts = &p.dsl;
    CHECK(vkAllocateDescriptorSets(dev, &dai, &set));
    VkDescriptorBufferInfo infos[2] = {{src, 0, src_size}, {dst, 0, dst_size}};
    VkWriteDescriptorSet writes[2] = {};
    for (int i = 0; i < 2; i++) {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = set;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &infos[i];
    }
    vkUpdateDescriptorSets(dev, 2, writes, 0, nullptr);
    return set;
}

static const uint32_t THREADS = 256;
static const uint32_t GROUPS = 1024;

// globals referenced by run_once (single-threaded probe)
static void* staging_src_data = nullptr;
static VkBuffer staging_dst_buf = VK_NULL_HANDLE;

struct Sample {
    double wall_ns;  // record+submit+wait (GPU timestamps unusable on this MoltenVK/AMD stack)
    double cpu_ns;   // memcpy / flush cost, 0 if none
};

// mode: 0=vram, 1=staging, 2=host-import, 3=managed
static Sample run_once(Device& d, Pipeline& p, int mode, VkDescriptorSet set,
                       Buf& staging, Buf& managed, VkDeviceSize bytes,
                       uint32_t nvec) {
    Sample s{};
    uint64_t t0 = now_ns();
    if (mode == 1) {
        uint64_t c0 = now_ns();
        memcpy(staging.mem.mapped, staging_src_data, bytes);
        s.cpu_ns = double(now_ns() - c0);
    } else if (mode == 3) {
        uint64_t c0 = now_ns();
        VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
        range.memory = managed.mem.mem;
        range.size = bytes;
        CHECK(vkFlushMappedMemoryRanges(d.dev, 1, &range));
        s.cpu_ns = double(now_ns() - c0);
    }

    CHECK(vkResetCommandBuffer(d.cb, 0));
    VkCommandBufferBeginInfo cbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    cbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    CHECK(vkBeginCommandBuffer(d.cb, &cbi));
    vkCmdBindPipeline(d.cb, VK_PIPELINE_BIND_POINT_COMPUTE, p.pipe);
    vkCmdBindDescriptorSets(d.cb, VK_PIPELINE_BIND_POINT_COMPUTE, p.pl, 0, 1, &set, 0, nullptr);
    vkCmdPushConstants(d.cb, p.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, 4, &nvec);
    if (mode == 1) {
        VkBufferCopy region{0, 0, bytes};
        vkCmdCopyBuffer(d.cb, staging.buf, staging_dst_buf, 1, &region);
        VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(d.cb, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
    }
    vkCmdDispatch(d.cb, GROUPS, 1, 1);
    CHECK(vkEndCommandBuffer(d.cb));

    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &d.cb;
    CHECK(vkResetFences(d.dev, 1, &d.fence));
    CHECK(vkQueueSubmit(d.queue, 1, &si, d.fence));
    CHECK(vkWaitForFences(d.dev, 1, &d.fence, VK_TRUE, UINT64_MAX));
    s.wall_ns = double(now_ns() - t0);
    return s;
}

static bool verify(Device& d, Pipeline& p, VkDescriptorSet set, Buf& out,
                   uint32_t nvec, uint64_t expect) {
    CHECK(vkResetCommandBuffer(d.cb, 0));
    VkCommandBufferBeginInfo cbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    CHECK(vkBeginCommandBuffer(d.cb, &cbi));
    vkCmdBindPipeline(d.cb, VK_PIPELINE_BIND_POINT_COMPUTE, p.pipe);
    vkCmdBindDescriptorSets(d.cb, VK_PIPELINE_BIND_POINT_COMPUTE, p.pl, 0, 1, &set, 0, nullptr);
    vkCmdPushConstants(d.cb, p.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, 4, &nvec);
    vkCmdDispatch(d.cb, GROUPS, 1, 1);
    VkMemoryBarrier hb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    hb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    hb.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(d.cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &hb, 0, nullptr, 0, nullptr);
    CHECK(vkEndCommandBuffer(d.cb));
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &d.cb;
    CHECK(vkResetFences(d.dev, 1, &d.fence));
    CHECK(vkQueueSubmit(d.queue, 1, &si, d.fence));
    CHECK(vkWaitForFences(d.dev, 1, &d.fence, VK_TRUE, UINT64_MAX));
    uint32_t* sums = static_cast<uint32_t*>(out.mem.mapped);
    uint64_t got = 0;
    for (uint32_t i = 0; i < GROUPS * THREADS; i++) got += sums[i];
    if (got != expect)
        fprintf(stderr, "verify detail: got=%llu expect=%llu\n",
                (unsigned long long)got, (unsigned long long)expect);
    return got == expect;
}

static double gbps(double ns, size_t bytes) { return double(bytes) / ns; }

int main(int argc, char** argv) {
    int rounds = 3;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--rounds") && i + 1 < argc) rounds = atoi(argv[++i]);
    }

    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.apiVersion = VK_API_VERSION_1_1;
    const char* inst_exts[] = {"VK_KHR_portability_enumeration"};
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    ici.flags = VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
    ici.enabledExtensionCount = 1;
    ici.ppEnabledExtensionNames = inst_exts;
    VkInstance inst;
    CHECK(vkCreateInstance(&ici, nullptr, &inst));

    uint32_t nd = 0;
    CHECK(vkEnumeratePhysicalDevices(inst, &nd, nullptr));
    std::vector<VkPhysicalDevice> pds(nd);
    CHECK(vkEnumeratePhysicalDevices(inst, &nd, pds.data()));

    const size_t sizes[] = {4096, 65536, 1 << 20, 16 << 20, 256 << 20};
    const int reps_small = 200, reps_mid = 60, reps_big = 20;
    const char* mode_names[] = {"vram", "staging", "host-import", "managed"};

    for (VkPhysicalDevice pd : pds) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(pd, &props);
        printf("\n==== %s ====\n", props.deviceName);

        uint32_t nq = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(pd, &nq, nullptr);
        std::vector<VkQueueFamilyProperties> qfps(nq);
        vkGetPhysicalDeviceQueueFamilyProperties(pd, &nq, qfps.data());
        uint32_t qfam = UINT32_MAX;
        for (uint32_t i = 0; i < nq; i++)
            if (qfps[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { qfam = i; break; }
        if (qfam == UINT32_MAX) { printf("no compute queue, skip\n"); continue; }

        uint32_t ne = 0;
        vkEnumerateDeviceExtensionProperties(pd, nullptr, &ne, nullptr);
        std::vector<VkExtensionProperties> exts(ne);
        vkEnumerateDeviceExtensionProperties(pd, nullptr, &ne, exts.data());
        auto has_ext = [&](const char* n) {
            for (auto& e : exts) if (!strcmp(e.extensionName, n)) return true;
            return false;
        };
        bool host_import_ok = has_ext("VK_EXT_external_memory_host");
        if (!host_import_ok) printf("NOTE: no VK_EXT_external_memory_host, mode 2 skipped\n");

        float prio = 1.0f;
        VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        qci.queueFamilyIndex = qfam;
        qci.queueCount = 1;
        qci.pQueuePriorities = &prio;
        const char* dev_exts[] = {"VK_EXT_external_memory_host"};
        VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
        dci.queueCreateInfoCount = 1;
        dci.pQueueCreateInfos = &qci;
        if (host_import_ok) {
            dci.enabledExtensionCount = 1;
            dci.ppEnabledExtensionNames = dev_exts;
        }
        Device d;
        d.pd = pd;
        CHECK(vkCreateDevice(pd, &dci, nullptr, &d.dev));
        d.qfam = qfam;
        vkGetDeviceQueue(d.dev, qfam, 0, &d.queue);
        vkGetPhysicalDeviceMemoryProperties(pd, &d.mp);
        d.ts_period = props.limits.timestampPeriod;

        VkCommandPoolCreateInfo cpci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        cpci.queueFamilyIndex = qfam;
        cpci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        CHECK(vkCreateCommandPool(d.dev, &cpci, nullptr, &d.pool));
        VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        cbai.commandPool = d.pool;
        cbai.commandBufferCount = 1;
        CHECK(vkAllocateCommandBuffers(d.dev, &cbai, &d.cb));
        VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        CHECK(vkCreateFence(d.dev, &fci, nullptr, &d.fence));

        Pipeline p = make_pipeline(d.dev);

        uint32_t t_private = find_type(pd, UINT32_MAX, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 1);
        uint32_t t_staging = find_type(pd, UINT32_MAX,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, -1);
        uint32_t t_managed = find_type(pd, UINT32_MAX,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, 1);
        // host import must land on coherent host-visible type (type1 per report)
        uint32_t t_host = find_type(pd, UINT32_MAX,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, -1);
        printf("mem types: private=%u staging=%u managed=%u host=%u\n",
               t_private, t_staging, t_managed, t_host);

        const size_t max_bytes = sizes[4];
        Buf dev_buf = make_buf(d.dev, pd, max_bytes, t_private, false);
        Buf staging = make_buf(d.dev, pd, max_bytes, t_staging, true);
        Buf managed = make_buf(d.dev, pd, max_bytes, t_managed, true);
        Buf out = make_buf(d.dev, pd, GROUPS * THREADS * 4, t_staging, true);

        void* host_mem = nullptr;
        Buf host_buf;
        if (host_import_ok) {
            if (posix_memalign(&host_mem, 4096, max_bytes)) { fprintf(stderr, "posix_memalign fail\n"); exit(1); }
            host_buf = make_buf(d.dev, pd, max_bytes, t_host, false, host_mem);
        }

        std::vector<uint8_t> pattern(max_bytes);
        fill_pattern(pattern.data(), max_bytes);
        memcpy(staging.mem.mapped, pattern.data(), max_bytes);
        memcpy(managed.mem.mapped, pattern.data(), max_bytes);
        if (host_mem) memcpy(host_mem, pattern.data(), max_bytes);
        VkMappedMemoryRange mr{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
        mr.memory = managed.mem.mem;
        mr.size = VK_WHOLE_SIZE;
        CHECK(vkFlushMappedMemoryRanges(d.dev, 1, &mr));
        // one-time upload of pattern into VRAM for baseline + staging dst
        {
            CHECK(vkResetCommandBuffer(d.cb, 0));
            VkCommandBufferBeginInfo cbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
            CHECK(vkBeginCommandBuffer(d.cb, &cbi));
            VkBufferCopy region{0, 0, max_bytes};
            vkCmdCopyBuffer(d.cb, staging.buf, dev_buf.buf, 1, &region);
            CHECK(vkEndCommandBuffer(d.cb));
            VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
            si.commandBufferCount = 1;
            si.pCommandBuffers = &d.cb;
            CHECK(vkResetFences(d.dev, 1, &d.fence));
            CHECK(vkQueueSubmit(d.queue, 1, &si, d.fence));
            CHECK(vkWaitForFences(d.dev, 1, &d.fence, VK_TRUE, UINT64_MAX));
        }

        staging_src_data = pattern.data();
        staging_dst_buf = dev_buf.buf;

        for (size_t bytes : sizes) {
            CHECK(vkResetDescriptorPool(d.dev, p.dpool, 0));
            uint32_t nvec = uint32_t(bytes / 16);
            uint64_t expect = expected_sum(pattern.data(), bytes);
            int reps = bytes <= (1 << 20) ? reps_small : (bytes <= (16 << 20) ? reps_mid : reps_big);

            VkDescriptorSet sets[4];
            bool mode_ok[4] = {true, true, host_import_ok, true};
            VkBuffer srcs[4] = {dev_buf.buf, dev_buf.buf,
                                host_import_ok ? host_buf.buf : dev_buf.buf, managed.buf};
            for (int m = 0; m < 4; m++) {
                if (!mode_ok[m]) continue;
                sets[m] = bind_set(d.dev, p, srcs[m], max_bytes, out.buf, GROUPS * THREADS * 4);
                memset(out.mem.mapped, 0, GROUPS * THREADS * 4);
                bool ok = verify(d, p, sets[m], out, nvec, expect);
                printf("size=%zu mode=%s verify=%s\n", bytes, mode_names[m], ok ? "MATCH" : "MISMATCH");
                if (!ok) mode_ok[m] = false;
            }

            std::vector<std::vector<Sample>> samples(4);
            for (int r = 0; r < rounds; r++) {
                for (int m = 0; m < 4; m++) {          // interleaved A/B per round
                    if (!mode_ok[m]) continue;
                    for (int i = 0; i < reps; i++)
                        samples[m].push_back(run_once(d, p, m, sets[m], staging, managed, bytes, nvec));
                }
            }

            printf("--- size=%zu reps=%d rounds=%d (min-of-%d, wall clock) ---\n", bytes, reps, rounds, rounds);
            printf("%-12s %12s %12s %12s\n", "mode", "e2e GB/s", "e2e us/op", "cpu us/op");
            for (int m = 0; m < 4; m++) {
                if (!mode_ok[m] || samples[m].empty()) continue;
                double wall = 1e30, cpu = 1e30;
                for (auto& s : samples[m]) {
                    wall = std::min(wall, s.wall_ns);
                    cpu = std::min(cpu, s.cpu_ns);
                }
                printf("%-12s %12.2f %12.2f %12.2f\n", mode_names[m],
                       gbps(wall, bytes), wall / 1000.0, cpu / 1000.0);
            }
            fflush(stdout);
        }

        if (host_mem) free(host_mem);
        vkDestroyDevice(d.dev, nullptr);
    }
    vkDestroyInstance(inst, nullptr);
    return 0;
}