// Image-layout accounting probe.
//
// The image backend stores every >=3-D tensor as a RGBA image2DArray
// (extent = {W, N*H, 1}, arrayLayers = ceil(C/4), see Tensor::getGPUShape).
// Tensor::getImageSize()/getGPUShape() only ever see the *compact* byte count,
// while the driver lays the array out with its own rowPitch and arrayPitch
// granularity. This tool measures that gap per tensor shape and reports what
// the same data would cost as a single folded image2D (z folded into x).
//
// It creates VkImages without binding memory (legal for
// vkGetImageMemoryRequirements and vkGetImageSubresourceLayout), so it does not
// disturb the runtime's allocations and never touches device memory.
//
// Usage:
//   tests/ImageLayoutProbe                          # synthetic shapes only
//   tests/ImageLayoutProbe path/to/model.vkopbin    # + real model tensors

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <typeinfo>
#include <vector>

#include "core/Tensor.hpp"
#include "core/runtime.hpp"
#include "include/logger.hpp"
#include "vulkan/VulkanDevice.hpp"
#include "vulkan/VulkanInstance.hpp"
#include "vulkan/VulkanLib.hpp"

using vkop::core::ITensor;
using vkop::core::Runtime;
using vkop::VulkanCommandPool;
using vkop::VulkanDevice;
using vkop::VulkanInstance;

namespace {

VkDevice g_device = VK_NULL_HANDLE;
VkPhysicalDevice g_physdev = VK_NULL_HANDLE;
uint32_t g_max_dim_2d = 0;
uint32_t g_max_layers = 0;

void *proc(const char *name) {
    // Vulkan is loaded via dlopen in VulkanLib; the loader's exported entry
    // points (global + device commands) are resolved through its dlsym handle.
    return vkop::VulkanLib::getVulkanLib().get_proc_address(name);
}

// 0 = success, otherwise the VkResult of vkCreateImage.
int try_create_image(const VkImageCreateInfo &ci, VkImage *out) {
    auto create = reinterpret_cast<PFN_vkCreateImage>(proc("vkCreateImage"));
    return static_cast<int>(create(g_device, &ci, nullptr, out));
}

struct Layout {
    VkImage image = VK_NULL_HANDLE;
    uint64_t size = 0;       // driver-reported total allocation
    uint64_t row_pitch = 0;  // bytes per row
    uint64_t array_pitch = 0; // bytes per array layer
    uint64_t size_1layer = 0; // same extent with arrayLayers = 1
    bool valid = false;
};

Layout measure(VkExtent3D extent, uint32_t layers, VkFormat format) {
    Layout l;
    VkImageCreateInfo ci = {};
    ci.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ci.imageType = VK_IMAGE_TYPE_2D;
    ci.format = format;
    ci.extent = extent;
    ci.mipLevels = 1;
    ci.arrayLayers = layers;
    ci.samples = VK_SAMPLE_COUNT_1_BIT;
    ci.tiling = VK_IMAGE_TILING_OPTIMAL;
    ci.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
               VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (try_create_image(ci, &l.image) != VK_SUCCESS) {
        return l;
    }
    l.valid = true;

    auto get_req = reinterpret_cast<PFN_vkGetImageMemoryRequirements>(proc(
        "vkGetImageMemoryRequirements"));
    VkMemoryRequirements req = {};
    get_req(g_device, l.image, &req);
    l.size = req.size;

    auto get_layout = reinterpret_cast<PFN_vkGetImageSubresourceLayout>(
        proc("vkGetImageSubresourceLayout"));
    VkSubresourceLayout sl0 = {};
    VkImageSubresource sr0 = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0};
    get_layout(g_device, l.image, &sr0, &sl0);
    l.row_pitch = sl0.rowPitch;
    l.array_pitch = sl0.arrayPitch;
    if (layers > 1) {
        VkSubresourceLayout sl1 = {};
        VkImageSubresource sr1 = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1};
        get_layout(g_device, l.image, &sr1, &sl1);
        if (sl1.offset > sl0.offset) {
            l.array_pitch = sl1.offset - sl0.offset;
        }
    }
    // vkGetImageSubresourceLayout returns implementation-defined (often zero)
    // values for OPTIMAL tiling. Fall back to the reliable vkGetImage...
    // Requirements total so the effective per-layer / per-row footprint is
    // still reported.
    if (l.array_pitch == 0 && layers > 0) {
        l.array_pitch = l.size / layers;
    }
    if (l.row_pitch == 0 && extent.height > 0) {
        l.row_pitch = l.array_pitch / extent.height;
    }

    if (layers > 1) {
        // Same footprint for a single slice tells us what one row-aligned
        // plane costs; the folded variant is computed by the caller.
        ci.arrayLayers = 1;
        VkImage one = VK_NULL_HANDLE;
        if (try_create_image(ci, &one) == VK_SUCCESS) {
            get_req(g_device, one, &req);
            l.size_1layer = req.size;
            reinterpret_cast<PFN_vkDestroyImage>(proc("vkDestroyImage"))(
                g_device, one, nullptr);
        }
    }
    return l;
}

void release(const Layout &l) {
    if (l.valid) {
        reinterpret_cast<PFN_vkDestroyImage>(proc("vkDestroyImage"))(
            g_device, l.image, nullptr);
    }
}

VkFormat format_for_bytes(int bytes_per_element) {
    switch (bytes_per_element) {
    case 1:
        return VK_FORMAT_R8G8B8A8_SNORM;
    case 2:
        return VK_FORMAT_R16G16B16A16_SFLOAT;
    default:
        return VK_FORMAT_R32G32B32A32_SFLOAT;
    }
}

int bytes_for_format(VkFormat f) {
    return (f == VK_FORMAT_R8G8B8A8_SNORM) ? 1 : (f == VK_FORMAT_R16G16B16A16_SFLOAT ? 2 : 4);
}

// Mirror of Tensor::getGPUShape() + make_vkimg() for the two packing modes.
enum class Role {
    Activation, // runtime tensor: folded layout has NO shader support today
    Kernel1x1,  // 1x1 conv weight: conv1x1()/pack path already exists
    Kernel3x3,  // kxk conv weight: would need a new folded shader branch
};

struct Shape {
    std::string name;
    int n, c, h, w;
    bool pack;     // fold C4 into width (1x1 conv weights)
    bool transpose; // conv weight layout: layers = C_in/4
    Role role = Role::Activation;
};

struct Row {
    std::string name;
    uint64_t compact = 0;
    uint64_t array_size = 0;
    uint64_t folded_size = 0;
    int width = 0, height = 0, layers = 0;
    int folded_width = 0, folded_height = 0;
    bool folded_ok = false;
    bool array_ok = false;
    Role role = Role::Activation;
};

// Fold decision: fold only when it saves both an absolute and a relative
// amount AND the folded extent fits device limits. `shaders_support` mirrors
// what the image shaders can actually address today (see conv2d.comp: only the
// 1x1 kernel pack path folds; activations/kxk kernels still index by gid.z).
struct Decision {
    bool want_fold = false;
    bool shaders_support = false;
    bool fits = false;
    uint64_t saves = 0;
    double pct = 0.0;
};

Decision decide_fold(const Row &r) {
    Decision d;
    d.fits = r.folded_ok;
    if (!r.array_ok || !r.folded_ok) {
        return d;
    }
    if (r.array_size > r.folded_size) {
        d.saves = r.array_size - r.folded_size;
        d.pct = 100.0 * (double)d.saves / (double)r.array_size;
    }
    d.want_fold = d.fits && d.saves >= 32u * 1024u && d.pct >= 10.0;
    d.shaders_support = (r.role == Role::Kernel1x1);
    return d;
}

Row measure_shape(const Shape &s, VkFormat format, bool verbose) {
    const int bpe = bytes_for_format(format);
    int batch = s.transpose ? s.c : s.n;  // folds into height
    int chan = s.transpose ? s.n : s.c;   // packs into RGBA quads
    int chan4 = (chan + 3) / 4;
    Row r;
    r.name = s.name;
    r.role = s.role;
    r.compact = static_cast<uint64_t>(s.w) * (s.h * batch) * chan4 * 4 * bpe;

    int width = s.w;
    int layers = chan4;
    if (s.pack) {
        width = s.w * chan4;
        layers = 1;
    }
    r.width = width;
    r.height = s.h * batch;
    r.layers = layers;

    Layout arr = measure(VkExtent3D{static_cast<uint32_t>(width),
                                    static_cast<uint32_t>(r.height), 1},
                         static_cast<uint32_t>(std::max(layers, 1)), format);
    r.array_ok = arr.valid;
    r.array_size = arr.valid ? arr.size : 0;
    if (verbose && arr.valid) {
        printf("      rowPitch=%llu arrayPitch=%llu singleLayer=%llu\n",
               (unsigned long long)arr.row_pitch,
               (unsigned long long)arr.array_pitch,
               (unsigned long long)arr.size_1layer);
    }
    release(arr);

    // Folded variant: z merged into x, one image2D plane.
    r.folded_width = s.w * chan4;
    r.folded_height = s.h * batch;
    if (r.folded_width > 0 &&
        static_cast<uint32_t>(r.folded_width) <= g_max_dim_2d &&
        static_cast<uint32_t>(r.folded_height) <= g_max_dim_2d) {
        Layout f = measure(
            VkExtent3D{static_cast<uint32_t>(r.folded_width),
                       static_cast<uint32_t>(r.folded_height), 1},
            1, format);
        r.folded_ok = f.valid;
        r.folded_size = f.valid ? f.size : 0;
        release(f);
    }
    return r;
}

void print_row(const Row &r) {
    double waste = r.array_ok && r.compact
                       ? 100.0 * (double)(r.array_size - r.compact) /
                             (double)r.compact
                       : 0.0;
    char folded[32];
    if (r.folded_ok) {
        snprintf(folded, sizeof(folded), "%lluKB",
                 (unsigned long long)(r.folded_size / 1024));
    } else {
        snprintf(folded, sizeof(folded), "n/a");
    }
    printf("  %-26s %5dx%-6d L=%-5d  compact=%-9llu array=%-9llu (%+6.2f%%)  "
           "folded2D=%-9s\n",
           r.name.c_str(), r.width, r.height, r.layers,
           (unsigned long long)r.compact, (unsigned long long)r.array_size,
           waste, folded);
    if (r.array_ok) {
        long long delta = (long long)r.array_size - (long long)r.folded_size;
        printf("  %-26s   folded width=%d (limit %u)%s, saves %lld KB vs array\n",
               "", r.folded_width, g_max_dim_2d,
               r.folded_ok ? "" : " -> OVER LIMIT, not foldable",
               (long long)(delta / 1024));
    }
    Decision d = decide_fold(r);
    const char *role = r.role == Role::Kernel1x1   ? "kernel1x1(shaders OK)"
                       : r.role == Role::Kernel3x3 ? "kernel3x3(NEEDS shader)"
                                                   : "activation(NEEDS shader)";
    const char *verdict;
    if (!d.want_fold) {
        verdict = "KEEP array (savings below threshold)";
    } else if (d.shaders_support) {
        verdict = "FOLD NOW (shader path exists)";
    } else {
        verdict = "FOLD candidate -> needs folded shader branch";
    }
    printf("  %-26s   [%s] -> %s\n", "", role, verdict);
}

int element_bytes_of(const std::shared_ptr<ITensor> &t) {
    const std::type_info &d = t->dtype();
    if (d == typeid(float)) {
        return 4;
    }
    if (d == typeid(uint16_t)) {
        return 2;
    }
    return 1;
}

void report_device_limits() {
    VkPhysicalDeviceProperties2 props2 = {};
    props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    auto get_props =
        reinterpret_cast<PFN_vkGetPhysicalDeviceProperties2>(
            proc("vkGetPhysicalDeviceProperties2"));
    get_props(g_physdev, &props2);
    const VkPhysicalDeviceProperties &props = props2.properties;
    printf("device            : %s\n", props.deviceName);
    printf("apiVersion        : %u.%u.%u\n",
           VK_API_VERSION_MAJOR(props.apiVersion),
           VK_API_VERSION_MINOR(props.apiVersion),
           VK_API_VERSION_PATCH(props.apiVersion));
    printf("maxImageDimension2D       : %u\n", props.limits.maxImageDimension2D);
    printf("maxImageArrayLayers       : %u\n", props.limits.maxImageArrayLayers);
    printf("maxMemoryAllocationCount  : %u\n",
           props.limits.maxMemoryAllocationCount);
    printf("bufferImageGranularity    : %llu\n",
           (unsigned long long)props.limits.bufferImageGranularity);
    const uint32_t probe_fmts[] = {VK_FORMAT_R32G32B32A32_SFLOAT,
                                   VK_FORMAT_R16G16B16A16_SFLOAT,
                                   VK_FORMAT_R8G8B8A8_SNORM};
    auto get_fmt =
        reinterpret_cast<PFN_vkGetPhysicalDeviceImageFormatProperties>(
            proc("vkGetPhysicalDeviceImageFormatProperties"));
    for (uint32_t fi : probe_fmts) {
        VkFormat f = static_cast<VkFormat>(fi);
        VkImageFormatProperties fp = {};
        auto res = get_fmt(g_physdev, f, VK_IMAGE_TYPE_2D,
                           VK_IMAGE_TILING_OPTIMAL,
                           VK_IMAGE_USAGE_SAMPLED_BIT |
                               VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                               VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                           0, &fp);
        if (res == VK_SUCCESS) {
            printf("  fmt %d: max extent %ux%u, max layers %u, maxResource %llu\n",
                   (int)f, fp.maxExtent.width, fp.maxExtent.height,
                   fp.maxArrayLayers, (unsigned long long)fp.maxResourceSize);
        } else {
            printf("  fmt %d: unsupported (%d)\n", (int)f, (int)res);
        }
    }
    g_max_dim_2d = props.limits.maxImageDimension2D;
    g_max_layers = props.limits.maxImageArrayLayers;
}

void report_synthetic() {
    // Faithful resnet18 shape inventory (batch=1, 224x224) so the fold
    // heuristic is exercised across a whole real CNN without a 130MB download.
    const Shape shapes[] = {
        // stem + stage activations: [n, c, h, w]
        {"act stem 64x112x112", 1, 64, 112, 112, false, false, Role::Activation},
        {"act 64x56x56", 1, 64, 56, 56, false, false, Role::Activation},
        {"act 128x28x28", 1, 128, 28, 28, false, false, Role::Activation},
        {"act 256x28x28", 1, 256, 28, 28, false, false, Role::Activation},
        {"act 256x14x14", 1, 256, 14, 14, false, false, Role::Activation},
        {"act 512x14x14", 1, 512, 14, 14, false, false, Role::Activation},
        {"act 512x7x7", 1, 512, 7, 7, false, false, Role::Activation},
        {"act 2048x7x7", 1, 2048, 7, 7, false, false, Role::Activation},
        // conv weights [cout, cin, kh, kw], transpose layout:
        //   width=k_w, height=k_h*cin, layers=cout/4 (grouped per shader).
        {"w 3x3 64->64", 64, 64, 3, 3, false, true, Role::Kernel3x3},
        {"w 3x3 128->128", 128, 128, 3, 3, false, true, Role::Kernel3x3},
        {"w 3x3 256->256", 256, 256, 3, 3, false, true, Role::Kernel3x3},
        {"w 3x3 512->512", 512, 512, 3, 3, false, true, Role::Kernel3x3},
        {"w 1x1 64->256 (proj)", 256, 64, 1, 1, false, true, Role::Kernel1x1},
        {"w 1x1 256->64 (se)", 64, 256, 1, 1, false, true, Role::Kernel1x1},
        {"w 1x1 512->256 (se)", 256, 512, 1, 1, false, true, Role::Kernel1x1},
        // final head: FC-as-1x1 over 2048 channels at 1x1 spatial.
        {"act 2048x1x1 (pool)", 1, 2048, 1, 1, false, false, Role::Activation},
    };
    for (VkFormat format : {VK_FORMAT_R32G32B32A32_SFLOAT,
                            VK_FORMAT_R16G16B16A16_SFLOAT}) {
        printf("\n== synthetic shapes (%s) ==\n",
               format == VK_FORMAT_R32G32B32A32_SFLOAT ? "fp32" : "fp16");
        uint64_t now = 0, needs = 0;
        int cnt_now = 0, cnt_needs = 0;
        for (const auto &s : shapes) {
            Row r = measure_shape(s, format, false);
            print_row(r);
            Decision d = decide_fold(r);
            if (d.want_fold) {
                if (d.shaders_support) {
                    now += d.saves;
                    ++cnt_now;
                } else {
                    needs += d.saves;
                    ++cnt_needs;
                }
            }
        }
        printf("  -- resnet18 sweep: FOLD NOW %d tensors (%.1f KB) | "
               "FOLD but needs shader %d tensors (%.1f KB)\n",
               cnt_now, (double)now / 1024.0, cnt_needs, (double)needs / 1024.0);
    }
}

void report_model(const std::string &path, const std::shared_ptr<VulkanCommandPool> &cmdpool) {
    printf("\n== model %s ==\n", path.c_str());
    auto rt = std::make_shared<Runtime>(cmdpool, path, 0);
    rt->LoadModel();
    auto tensors = rt->ListTensors();
    std::sort(tensors.begin(), tensors.end(),
              [](const std::pair<std::string, std::shared_ptr<ITensor>> &a,
                 const std::pair<std::string, std::shared_ptr<ITensor>> &b) {
                  return a.first < b.first;
              });

    uint64_t total_compact = 0, total_array = 0, total_folded = 0;
    uint64_t foldable_now = 0, foldable_needs_shader = 0;
    int fold_now_cnt = 0, needs_shader_cnt = 0;
    int counted = 0, skipped_lowrank = 0, too_many_layers = 0;
    std::vector<Row> rows;
    for (const auto &kv : tensors) {
        auto t = kv.second;
        if (!t || t->num_dims() < 3) {
            ++skipped_lowrank;
            continue;
        }
        auto dims = t->getShape();
        int nd = dims.size();
        Shape s{};
        s.name = kv.first;
        s.n = (nd == 4) ? dims[0] : 1;
        s.c = dims[nd - 3];
        s.h = dims[nd - 2];
        s.w = dims[nd - 1];
        s.pack = t->get_pack();
        s.transpose = t->get_transpose();
        // The runtime sets transpose_ only on Conv weights and pack_ only on
        // 1x1 Conv weights (runtime.cpp:146), so the flags identify the role.
        if (s.pack) {
            s.role = Role::Kernel1x1;
        } else if (s.transpose) {
            s.role = Role::Kernel3x3;
        } else {
            s.role = Role::Activation;
        }
        int chan = s.transpose ? s.n : s.c;
        if ((chan + 3) / 4 > (int)g_max_layers) {
            ++too_many_layers;
            printf("  %-40s SKIPPED: needs %d layers > maxImageArrayLayers %u\n",
                   s.name.c_str(), (chan + 3) / 4, g_max_layers);
            continue;
        }
        auto r = measure_shape(s, format_for_bytes(element_bytes_of(t)), false);
        rows.push_back(r);
        total_compact += r.compact;
        total_array += r.array_size;
        if (r.folded_ok) {
            total_folded += r.folded_size;
        }
        Decision d = decide_fold(r);
        if (d.want_fold) {
            if (d.shaders_support) {
                foldable_now += d.saves;
                ++fold_now_cnt;
            } else {
                foldable_needs_shader += d.saves;
                ++needs_shader_cnt;
            }
        }
        ++counted;
    }
    for (const auto &r : rows) {
        print_row(r);
    }
    std::sort(rows.begin(), rows.end(), [](const Row &a, const Row &b) {
        return (a.array_size - a.compact) > (b.array_size - b.compact);
    });
    printf("\n  worst %zu by absolute padding:\n", std::min<size_t>(15, rows.size()));
    for (size_t i = 0; i < rows.size() && i < 15; ++i) {
        const auto &r = rows[i];
        printf("    %-40s +%8.1f KB (%.1f%%)\n", r.name.c_str(),
               (double)(r.array_size - r.compact) / 1024.0,
               r.compact ? 100.0 * (double)(r.array_size - r.compact) /
                               (double)r.compact
                         : 0.0);
    }
    printf("\n  tensors on image backend : %d (low-rank/buffer-only: %d, "
           "layer-overflow: %d)\n",
           counted, skipped_lowrank, too_many_layers);
    printf("  compact %8.2f MB | array-allocated %8.2f MB (+%.2f MB, %+.2f%%) | "
           "folded-2D %8.2f MB\n",
           (double)total_compact / 1048576.0,
           (double)total_array / 1048576.0,
           (double)(total_array - total_compact) / 1048576.0,
           total_compact ? 100.0 * (double)(total_array - total_compact) /
                               (double)total_compact
                         : 0.0,
           (double)total_folded / 1048576.0);

    printf("\n  fold decision (threshold: >32KB AND >10%% AND width<=limit):\n");
    printf("    FOLD NOW (1x1 kernel, conv1x1/pack path exists): %d tensors, "
           "%.1f MB recoverable\n",
           fold_now_cnt, (double)foldable_now / 1048576.0);
    printf("    FOLD candidate but NEEDS a new shader branch (kxk kernel / "
           "activation): %d tensors, %.1f MB recoverable\n",
           needs_shader_cnt, (double)foldable_needs_shader / 1048576.0);
}

} // namespace

int main(int argc, char **argv) {
    Logger::getInstance().setLevel(LOG_ERROR);
    auto &inst = VulkanInstance::getVulkanInstance();
    auto &phydevs = inst.getPhysicalDevices();
    if (phydevs.empty()) {
        printf("no vulkan device (set VK_ICD_FILENAMES)\n");
        return 1;
    }
    auto dev = std::make_shared<VulkanDevice>(phydevs[0]);
    if (dev->getDeviceName().find("llvmpipe") != std::string::npos) {
        printf("no valid vulkan device\n");
        return 1;
    }
    g_device = dev->getLogicalDevice();
    g_physdev = dev->getPhysicalDevice();

    report_device_limits();
    report_synthetic();

    if (argc > 1) {
        auto cmdpool = std::make_shared<VulkanCommandPool>(dev);
        report_model(argv[1], cmdpool);
    } else {
        printf("\n(no model path given — pass a .vkopbin to measure real tensors)\n");
    }
    return 0;
}
