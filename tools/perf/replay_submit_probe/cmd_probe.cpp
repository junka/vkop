// Does re-submitting an already-recorded SIMULTANEOUS_USE command buffer cost
// less than re-recording it? vkop's segment replay assumes yes; on MoltenVK the
// opposite was measured (record 19ms -> 0.1ms, but submit 0.1ms -> 49-97ms).
//
// The probe reproduces the shape of one vkop decode round -- SEG segments, DISP
// dispatches each, one timeline semaphore per segment, every segment waiting the
// previous segment's signal, submissions flushed in batches of BATCH, then a
// per-segment vkWaitSemaphores tail (what Runtime::Run() ends with in graph
// mode) -- and varies only what happens to the command buffers between rounds.
//
// build/run: see README.md next to this file.
#include <vulkan/vulkan.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>

#define CHECK(x)                                                             \
    do {                                                                     \
        VkResult _r = (x);                                                   \
        if (_r != VK_SUCCESS) {                                              \
            std::fprintf(stderr, "%s -> %d at line %d\n", #x, _r, __LINE__); \
            std::exit(2);                                                    \
        }                                                                    \
    } while (0)

static int SEG = 375;          // segments per round (vkop replay: one per level)
static int DISP = 2;           // dispatches recorded per segment
static const int ROUNDS = 40;  // rounds measured (first 3 warm up)
static int BATCH = 32;         // vkop's VKOP_SEG_BATCH default

static uint32_t *g_counter = nullptr;

struct Seg {
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VkSemaphore sem = VK_NULL_HANDLE;
    uint64_t sigval = 0;
    std::vector<VkSemaphore> sigs, waits;
    std::vector<uint64_t> sigvals, waitvals;
    VkPipelineStageFlags stage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    VkTimelineSemaphoreSubmitInfo tsi{};
};

static std::vector<char> read_file(const char *p) {
    std::ifstream f(p, std::ios::binary);
    if (!f) {
        std::fprintf(stderr, "cannot open %s\n", p);
        std::exit(1);
    }
    return std::vector<char>((std::istreambuf_iterator<char>(f)),
                             std::istreambuf_iterator<char>());
}

int main(int argc, char **argv) {
    setbuf(stdout, nullptr);
    if (argc < 2) {
        std::fprintf(stderr,
                     "usage: %s <nop.spv> [segments=375] [dispatches=2] "
                     "[batch=32]\n",
                     argv[0]);
        return 1;
    }
    if (argc > 2) SEG = std::atoi(argv[2]);
    if (argc > 3) DISP = std::atoi(argv[3]);
    if (argc > 4) BATCH = std::atoi(argv[4]);
    auto spv = read_file(argv[1]);

    VkInstance inst = VK_NULL_HANDLE;
    std::vector<const char *> iext;
    {
        const char *want[] = {VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME,
                              VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME};
        uint32_t ne = 0;
        vkEnumerateInstanceExtensionProperties(nullptr, &ne, nullptr);
        std::vector<VkExtensionProperties> avail(ne);
        vkEnumerateInstanceExtensionProperties(nullptr, &ne, avail.data());
        for (auto &w : want) {
            for (auto &a : avail) {
                if (std::strcmp(a.extensionName, w) == 0) {
                    iext.push_back(w);
                    break;
                }
            }
        }
    }
    VkApplicationInfo ai{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    ai.pApplicationName = "cmd_probe";
    // The loader hands back NULL trampolines for entry points newer than the
    // instance's apiVersion, so vkGetSemaphoreCounterValue crashes if this is
    // left at the 1.0 default. (Same class of bug as 20b424e in vkop.)
    ai.apiVersion = VK_API_VERSION_1_2;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &ai;
    ici.enabledExtensionCount = (uint32_t)iext.size();
    ici.ppEnabledExtensionNames = iext.data();
    ici.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
    CHECK(vkCreateInstance(&ici, nullptr, &inst));
    std::printf("[probe] instance ok\n");

    uint32_t npd = 0;
    vkEnumeratePhysicalDevices(inst, &npd, nullptr);
    std::vector<VkPhysicalDevice> pds(npd);
    vkEnumeratePhysicalDevices(inst, &npd, pds.data());
    VkPhysicalDevice pd = pds[0];
    for (auto d : pds) {
        VkPhysicalDeviceProperties p;
        vkGetPhysicalDeviceProperties(d, &p);
        if (p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) pd = d;
    }
    std::printf("[probe] %d physical device(s)\n", npd);
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(pd, &props);
    std::printf("device: %s\n", props.deviceName);

    uint32_t nqf = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &nqf, nullptr);
    std::vector<VkQueueFamilyProperties> qf(nqf);
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &nqf, qf.data());
    uint32_t qfi = 0;
    for (uint32_t i = 0; i < nqf; ++i)
        if (qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { qfi = i; break; }

    const char *dext[] = {VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME};
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = qfi;
    qci.queueCount = 1;
    float prio = 1.0f;
    qci.pQueuePriorities = &prio;
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = 1;
    dci.ppEnabledExtensionNames = dext;
    VkDevice dev = VK_NULL_HANDLE;
    CHECK(vkCreateDevice(pd, &dci, nullptr, &dev));
    VkQueue queue = VK_NULL_HANDLE;
    vkGetDeviceQueue(dev, qfi, 0, &queue);

    VkCommandPool cpool = VK_NULL_HANDLE;
    VkCommandPoolCreateInfo cpci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    cpci.queueFamilyIndex = qfi;
    CHECK(vkCreateCommandPool(dev, &cpci, nullptr, &cpool));

    VkBuffer buf = VK_NULL_HANDLE;
    VkDeviceMemory mem = VK_NULL_HANDLE;
    {
        VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bci.size = 4096;
        bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        CHECK(vkCreateBuffer(dev, &bci, nullptr, &buf));
        VkMemoryRequirements mr;
        vkGetBufferMemoryRequirements(dev, buf, &mr);
        VkPhysicalDeviceMemoryProperties mp;
        vkGetPhysicalDeviceMemoryProperties(pd, &mp);
        uint32_t mi = 0;
        for (uint32_t i = 0; i < mp.memoryTypeCount; ++i)
            if ((mr.memoryTypeBits & (1u << i)) &&
                (mp.memoryTypes[i].propertyFlags &
                 (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                  VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) ==
                    (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                     VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
                mi = i;
                break;
            }
        VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        mai.allocationSize = mr.size;
        mai.memoryTypeIndex = mi;
        CHECK(vkAllocateMemory(dev, &mai, nullptr, &mem));
        CHECK(vkBindBufferMemory(dev, buf, mem, 0));
        void *mapped = nullptr;
        CHECK(vkMapMemory(dev, mem, 0, VK_WHOLE_SIZE, 0, &mapped));
        g_counter = (uint32_t *)mapped;
        g_counter[0] = 0;
        VkMappedMemoryRange fr{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
        fr.memory = mem;
        fr.offset = 0;
        fr.size = VK_WHOLE_SIZE;
        vkFlushMappedMemoryRanges(dev, 1, &fr);
    }

    VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
    VkDescriptorSet dset = VK_NULL_HANDLE;
    {
        VkDescriptorSetLayoutBinding b{};
        b.binding = 0;
        b.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        b.descriptorCount = 1;
        b.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        VkDescriptorSetLayoutCreateInfo lci{
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        lci.bindingCount = 1;
        lci.pBindings = &b;
        CHECK(vkCreateDescriptorSetLayout(dev, &lci, nullptr, &dsl));

        VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1};
        VkDescriptorPoolCreateInfo ppi{
            VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        ppi.maxSets = 1;
        ppi.poolSizeCount = 1;
        ppi.pPoolSizes = &ps;
        VkDescriptorPool pool = VK_NULL_HANDLE;
        CHECK(vkCreateDescriptorPool(dev, &ppi, nullptr, &pool));
        VkDescriptorSetAllocateInfo sai{
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        sai.descriptorPool = pool;
        sai.descriptorSetCount = 1;
        sai.pSetLayouts = &dsl;
        CHECK(vkAllocateDescriptorSets(dev, &sai, &dset));
        VkDescriptorBufferInfo dbi{buf, 0, VK_WHOLE_SIZE};
        VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w.dstSet = dset;
        w.dstBinding = 0;
        w.descriptorCount = 1;
        w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w.pBufferInfo = &dbi;
        vkUpdateDescriptorSets(dev, 1, &w, 0, nullptr);
    }

    VkPipeline pl = VK_NULL_HANDLE;
    VkPipelineLayout dlayout = VK_NULL_HANDLE;
    {
        VkShaderModuleCreateInfo smci{
            VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        smci.codeSize = spv.size();
        smci.pCode = reinterpret_cast<const uint32_t *>(spv.data());
        VkShaderModule sm = VK_NULL_HANDLE;
        CHECK(vkCreateShaderModule(dev, &smci, nullptr, &sm));
        VkPipelineLayoutCreateInfo plci{
            VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        plci.setLayoutCount = 1;
        plci.pSetLayouts = &dsl;
		CHECK(vkCreatePipelineLayout(dev, &plci, nullptr, &dlayout));
        VkComputePipelineCreateInfo cpi{
            VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        cpi.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        cpi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        cpi.stage.module = sm;
        cpi.stage.pName = "main";
        cpi.layout = dlayout;
        CHECK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpi, nullptr, &pl));
        vkDestroyShaderModule(dev, sm, nullptr);
    }

    std::printf("[probe] pipeline ok\n");
    std::vector<Seg> segs(SEG);
    for (auto &s : segs) {
        VkSemaphoreTypeCreateInfo stci{};
        stci.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
        stci.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
        stci.initialValue = 0;
        VkSemaphoreCreateInfo sci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        sci.pNext = &stci;
        CHECK(vkCreateSemaphore(dev, &sci, nullptr, &s.sem));
        VkCommandBufferAllocateInfo cai{
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        cai.commandPool = cpool;
        cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cai.commandBufferCount = 1;
        CHECK(vkAllocateCommandBuffers(dev, &cai, &s.cmd));
    }

    uint64_t next_val = 1;

    auto record_seg = [&](Seg &s, VkCommandBufferUsageFlags flags) {
        CHECK(vkResetCommandBuffer(s.cmd, 0));
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags = flags;
        CHECK(vkBeginCommandBuffer(s.cmd, &bi));
        vkCmdBindPipeline(s.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pl);
        vkCmdBindDescriptorSets(s.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                dlayout, 0, 1, &dset, 0, nullptr);
        for (int i = 0; i < DISP; ++i) vkCmdDispatch(s.cmd, 4, 1, 1);
        CHECK(vkEndCommandBuffer(s.cmd));
    };

    auto fill_si = [&](Seg &s, VkSemaphore prev_sem, uint64_t prev_val,
                       bool chain, VkSubmitInfo *out) {
        s.sigs.clear();
        s.sigvals.clear();
        s.waits.clear();
        s.waitvals.clear();
        s.sigval = next_val++;
        s.sigs.push_back(s.sem);
        s.sigvals.push_back(s.sigval);
        if (chain && prev_sem != VK_NULL_HANDLE) {
            s.waits.push_back(prev_sem);
            s.waitvals.push_back(prev_val);
        }
        s.tsi = VkTimelineSemaphoreSubmitInfo{};
        s.tsi.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
        s.tsi.signalSemaphoreValueCount = 1;
        s.tsi.pSignalSemaphoreValues = s.sigvals.data();
        s.tsi.waitSemaphoreValueCount = (uint32_t)s.waitvals.size();
        s.tsi.pWaitSemaphoreValues =
            s.waitvals.empty() ? nullptr : s.waitvals.data();

        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1;
        si.pCommandBuffers = &s.cmd;
        si.signalSemaphoreCount = 1;
        si.pSignalSemaphores = s.sigs.data();
        si.waitSemaphoreCount = (uint32_t)s.waits.size();
        si.pWaitSemaphores = s.waits.empty() ? nullptr : s.waits.data();
        si.pWaitDstStageMask = s.waits.empty() ? nullptr : &s.stage;
        si.pNext = &s.tsi;
        *out = si;
    };

    auto tail_wait = [&]() {
        for (auto &s : segs) {
            uint64_t cur = 0;
            vkGetSemaphoreCounterValue(dev, s.sem, &cur);
            if (cur > s.sigval) continue;
            VkSemaphoreWaitInfo wi{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
            wi.semaphoreCount = 1;
            wi.pSemaphores = &s.sem;
            wi.pValues = &s.sigval;
            CHECK(vkWaitSemaphores(dev, &wi, UINT64_MAX));
        }
    };

    const uint64_t PER_DISPATCH = 4ull * 64ull;   // workgroups x local_size
    uint64_t prev_counter = 0;
    int verify_failures = 0;
    auto verify_round = [&](size_t nseg) {
        const uint64_t want = (uint64_t)nseg * DISP * PER_DISPATCH;
        const uint64_t got = g_counter[0] - prev_counter;
        prev_counter = g_counter[0];
        if (got != want) {
            std::fprintf(stderr, "[verify] want=%llu got=%llu\n",
                         (unsigned long long)want, (unsigned long long)got);
            verify_failures++;
        }
    };
    auto arm_line = [&](const char *name, int fail_before, double rec,
                        double sub, double wt, double tot) {
        std::printf("  %-26s record=%7.2f  submit=%7.2f  tailwait=%7.2f  "
                    "total=%7.2f  %s\n",
                    name, rec, sub, wt, tot,
                    verify_failures == fail_before
                        ? "verified"
                        : "VERIFY-FAIL (work did not run)");
    };

    using clock = std::chrono::steady_clock;
    auto ms = [](clock::time_point a, clock::time_point b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };
    auto median = [](std::vector<double> v) {
        std::sort(v.begin(), v.end());
        return v.empty() ? 0.0 : v[v.size() / 2];
    };

    auto run = [&](const char *name, bool redo_record, bool sim_use, bool chain,
                   int batch) {
        int fail_before = verify_failures;
        std::vector<double> rc, sc, wc, tc;
        VkCommandBufferUsageFlags flags =
            sim_use ? VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT
                    : VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        for (int round = 0; round < ROUNDS; ++round) {
            double rec = 0, sub = 0, wt = 0;
            auto t0 = clock::now();
            std::vector<VkSubmitInfo> sis;
            sis.reserve(batch);
            auto flush = [&]() {
                if (sis.empty()) return;
                auto f0 = clock::now();
                CHECK(vkQueueSubmit(queue, (uint32_t)sis.size(), sis.data(),
                                    VK_NULL_HANDLE));
                sub += ms(f0, clock::now());
                sis.clear();
            };
            VkSemaphore prev_sem = VK_NULL_HANDLE;
            uint64_t prev_val = 0;
            for (int i = 0; i < SEG; ++i) {
                Seg &s = segs[i];
                auto r0 = clock::now();
                if (redo_record || round == 0) record_seg(s, flags);
                rec += ms(r0, clock::now());
                VkSubmitInfo si;
                fill_si(s, prev_sem, prev_val, chain, &si);
                prev_sem = s.sem;
                prev_val = s.sigval;
                sis.push_back(si);
                if ((int)sis.size() >= batch) flush();
            }
            flush();
            auto w0 = clock::now();
            tail_wait();
            wt = ms(w0, clock::now());
            verify_round(SEG);
            if (round >= 3) {
                rc.push_back(rec);
                sc.push_back(sub);
                wc.push_back(wt);
                tc.push_back(ms(t0, clock::now()));
            }
        }
        arm_line(name, fail_before, median(rc), median(sc), median(wc), median(tc));
    };

    // ---- secondary command buffer arms -------------------------------------
    //
    // A secondary stays in EXECUTABLE after vkEndCommandBuffer and after being
    // executed, unlike a primary, which goes pending on submit. That is the
    // property replay actually wants: record the segments once, then re-execute
    // them from one primary and submit that primary a single time per round --
    // no per-segment timeline semaphore, no per-segment vkQueueSubmit.
    std::vector<VkCommandBuffer> secs(SEG);
    {
        VkCommandBufferAllocateInfo cai{
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        cai.commandPool = cpool;
        cai.level = VK_COMMAND_BUFFER_LEVEL_SECONDARY;
        cai.commandBufferCount = (uint32_t)SEG;
        CHECK(vkAllocateCommandBuffers(dev, &cai, secs.data()));
    }
    auto record_secondary_once = [&]() {
        for (int i = 0; i < SEG; ++i) {
            VkCommandBufferInheritanceInfo inh{};
            inh.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_INFO;
            VkCommandBufferBeginInfo bi{
                VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
            bi.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
            bi.pInheritanceInfo = &inh;
            CHECK(vkBeginCommandBuffer(secs[i], &bi));
            vkCmdBindPipeline(secs[i], VK_PIPELINE_BIND_POINT_COMPUTE, pl);
            vkCmdBindDescriptorSets(secs[i], VK_PIPELINE_BIND_POINT_COMPUTE,
                                    dlayout, 0, 1, &dset, 0, nullptr);
            for (int d = 0; d < DISP; ++d) vkCmdDispatch(secs[i], 4, 1, 1);
            CHECK(vkEndCommandBuffer(secs[i]));
        }
    };
    VkCommandBuffer prim = VK_NULL_HANDLE;
    {
        VkCommandBufferAllocateInfo cai{
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        cai.commandPool = cpool;
        cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cai.commandBufferCount = 1;
        CHECK(vkAllocateCommandBuffers(dev, &cai, &prim));
    }
    VkSemaphore prim_sem = VK_NULL_HANDLE;
    {
        VkSemaphoreTypeCreateInfo stci{};
        stci.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
        stci.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
        VkSemaphoreCreateInfo sci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        sci.pNext = &stci;
        CHECK(vkCreateSemaphore(dev, &sci, nullptr, &prim_sem));
    }
    auto record_prim_exec = [&](VkCommandBufferUsageFlags flags) {
        CHECK(vkResetCommandBuffer(prim, 0));
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags = flags;
        CHECK(vkBeginCommandBuffer(prim, &bi));
        vkCmdExecuteCommands(prim, (uint32_t)SEG, secs.data());
        CHECK(vkEndCommandBuffer(prim));
    };
    auto run_secondary = [&](const char *name, bool re_record_primary) {
        int fail_before = verify_failures;
        std::vector<double> rc, sc, wc, tc;
        record_secondary_once();
        VkCommandBufferUsageFlags pflags =
            re_record_primary ? VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
                              : VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
        if (!re_record_primary) record_prim_exec(pflags);
        uint64_t pval = 0;
        for (int round = 0; round < ROUNDS; ++round) {
            double rec = 0, sub = 0, wt = 0;
            auto t0 = clock::now();
            if (re_record_primary) {
                auto r0 = clock::now();
                record_prim_exec(pflags);
                rec = ms(r0, clock::now());
            }
            pval = next_val++;
            VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
            si.commandBufferCount = 1;
            si.pCommandBuffers = &prim;
            si.signalSemaphoreCount = 1;
            si.pSignalSemaphores = &prim_sem;
            VkTimelineSemaphoreSubmitInfo tsi{};
            tsi.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
            tsi.signalSemaphoreValueCount = 1;
            tsi.pSignalSemaphoreValues = &pval;
            si.pNext = &tsi;
            auto s0 = clock::now();
            VkResult sr = vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE);
            sub = ms(s0, clock::now());
            if (sr != VK_SUCCESS) {
                std::fprintf(stderr, "%s: vkQueueSubmit -> %d\n", name, sr);
                return;
            }
            auto w0 = clock::now();
            uint64_t cur = 0;
            vkGetSemaphoreCounterValue(dev, prim_sem, &cur);
            if (cur <= pval) {
                VkSemaphoreWaitInfo wi{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
                wi.semaphoreCount = 1;
                wi.pSemaphores = &prim_sem;
                wi.pValues = &pval;
                CHECK(vkWaitSemaphores(dev, &wi, UINT64_MAX));
            }
            wt = ms(w0, clock::now());
            verify_round(SEG);
            if (round >= 3) {
                rc.push_back(rec);
                sc.push_back(sub);
                wc.push_back(wt);
                tc.push_back(ms(t0, clock::now()));
            }
        }
        arm_line(name, fail_before, median(rc), median(sc), median(wc),
                 median(tc));
    };

    // vkCmdExecuteCommands is the real API; the wrapper name below keeps the
    // call sites readable.
    // A round split across TWO primaries with a host wait in between is the
    // shape vkop's mid-round readback needs (the pre_readback_hook closes the
    // segment, submits what has been recorded so far, does a synchronous
    // D2H, then keeps recording). The LLM's decode rounds report zero readback
    // boundaries, so this submission shape is otherwise untested here.
    auto run_secondary_split = [&]() {
        int fail_before = verify_failures;
        std::vector<double> rc, sc, wc, tc;
        record_secondary_once();
        const int half = SEG / 2;
        VkCommandBuffer prim2 = VK_NULL_HANDLE;
        {
            VkCommandBufferAllocateInfo cai{
                VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
            cai.commandPool = cpool;
            cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            cai.commandBufferCount = 1;
            CHECK(vkAllocateCommandBuffers(dev, &cai, &prim2));
        }
        for (int round = 0; round < ROUNDS; ++round) {
            double rec = 0, sub = 0, wt = 0;
            auto t0 = clock::now();
            for (int part = 0; part < 2; ++part) {
                const int from = part == 0 ? 0 : half;
                const int n = part == 0 ? half : SEG - half;
                VkCommandBuffer p = part == 0 ? prim : prim2;
                VkCommandBuffer *list = secs.data() + from;
                auto r0 = clock::now();
                CHECK(vkResetCommandBuffer(p, 0));
                VkCommandBufferBeginInfo bi{
                    VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
                bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
                CHECK(vkBeginCommandBuffer(p, &bi));
                vkCmdExecuteCommands(p, (uint32_t)n, list);
                CHECK(vkEndCommandBuffer(p));
                rec += ms(r0, clock::now());
                uint64_t val = next_val++;
                VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
                si.commandBufferCount = 1;
                si.pCommandBuffers = &p;
                si.signalSemaphoreCount = 1;
                si.pSignalSemaphores = &prim_sem;
                VkTimelineSemaphoreSubmitInfo tsi{};
                tsi.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
                tsi.signalSemaphoreValueCount = 1;
                tsi.pSignalSemaphoreValues = &val;
                si.pNext = &tsi;
                auto s0 = clock::now();
                CHECK(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE));
                sub += ms(s0, clock::now());
                // The host reads back between the two halves, so this wait is
                // on the critical path, not just at the round boundary.
                auto w0 = clock::now();
                uint64_t cur = 0;
                vkGetSemaphoreCounterValue(dev, prim_sem, &cur);
                if (cur <= val) {
                    VkSemaphoreWaitInfo wi{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
                    wi.semaphoreCount = 1;
                    wi.pSemaphores = &prim_sem;
                    wi.pValues = &val;
                    CHECK(vkWaitSemaphores(dev, &wi, UINT64_MAX));
                }
                wt += ms(w0, clock::now());
            }
            verify_round(SEG);
            if (round >= 3) {
                rc.push_back(rec);
                sc.push_back(sub);
                wc.push_back(wt);
                tc.push_back(ms(t0, clock::now()));
            }
        }
        arm_line("secondary: split round x2", fail_before, median(rc),
                 median(sc), median(wc), median(tc));
    };

    std::printf("%d segments x %d dispatches, batch %d, median of %d rounds:\n",
                SEG, DISP, BATCH, ROUNDS - 3);
    run("re-record SIMULTANEOUS", true, true, true, BATCH);
    run("re-submit (replay-like)", false, true, true, BATCH);
    run("re-submit batch=1", false, true, true, 1);
    run("re-submit no chain", false, true, false, BATCH);
    run("re-record no chain", true, true, false, BATCH);
    run("re-record ONE_TIME", true, false, true, BATCH);
    run_secondary("secondary: re-exec primary", true);
    run_secondary("secondary: submit primary only", false);
    run_secondary_split();
    std::fflush(stdout);

    vkDeviceWaitIdle(dev);
    return 0;
}
