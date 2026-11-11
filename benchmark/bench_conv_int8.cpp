// int8 3x3 stride-1 Conv2d: element path vs the register-blocked path.
//
// The register-blocked path in shaders/buffer/conv2d.comp (conv_row3) also
// decodes int8 weights now, which is the shape real QDQ models use: the
// quantized convs in a typical CNN stem are 3x3 stride-1. This benchmark A/Bs
// the two paths on the SAME shape and data by toggling VKOP_CONV_ELEM_ONLY,
// which pins the element path in the host gate.
//
// Every shape is also checked against a CPU reference computed from the same
// quantized weights, so a path that decodes bytes differently (wrong sign
// extension, wrong scale placement) fails here rather than in a model.
//
// Run:
//   bench_conv_int8 [reps]                        # blocked path
//   VKOP_CONV_ELEM_ONLY=1 bench_conv_int8 [reps]  # element path
//
// Measured on Intel ARL (iGPU, so absolute numbers are power-limited and
// noisy; the ratio is the point): at 64 channels the blocked path is 1.3-2x
// the element path, at 32 channels the two are level because the per-thread
// work is too small to amortise the staging.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <vector>

#include "core/ShaderArchive.hpp"
#include "core/Tensor.hpp"
#include "include/dev_shaders.hpp"
#include "include/logger.hpp"
#include "ops/OperatorFactory.hpp"
#include "ops/Ops.hpp"
#include "vulkan/VulkanCommandPool.hpp"
#include "vulkan/VulkanDevice.hpp"
#include "vulkan/VulkanInstance.hpp"

using namespace vkop;

namespace {
struct Shape {
    int n, ic, ih, iw, oc, k, s, p, g;
    const char *name;
};

// Direct convolution reference on the dequantized weights, same tap order as
// the shader (ascending ic, ky, kx).
std::vector<float> ref_conv(const Shape &sh, const std::vector<float> &x,
                            const std::vector<int8_t> &w8,
                            const std::vector<float> &scale,
                            const std::vector<float> &bias) {
    const int oh = (sh.ih + 2 * sh.p - sh.k) / sh.s + 1;
    const int ow = (sh.iw + 2 * sh.p - sh.k) / sh.s + 1;
    const int icg = sh.ic / sh.g;
    const int ocg = sh.oc / sh.g;
    std::vector<float> y(static_cast<size_t>(sh.n) * sh.oc * oh * ow, 0.0f);
    for (int n = 0; n < sh.n; n++)
        for (int oc = 0; oc < sh.oc; oc++) {
            const int g = oc / ocg;
            const int ic_base = g * icg;
            for (int yy = 0; yy < oh; yy++)
                for (int xx = 0; xx < ow; xx++) {
                    double sum = 0.0;
                    for (int kic = 0; kic < icg; kic++)
                        for (int ky = 0; ky < sh.k; ky++) {
                            const int iy = yy * sh.s + ky - sh.p;
                            if (iy < 0 || iy >= sh.ih)
                                continue;
                            for (int kx = 0; kx < sh.k; kx++) {
                                const int ix = xx * sh.s + kx - sh.p;
                                if (ix < 0 || ix >= sh.iw)
                                    continue;
                                const size_t xi =
                                    ((static_cast<size_t>(n) * sh.ic + ic_base +
                                      kic) *
                                         sh.ih +
                                     iy) *
                                        sh.iw +
                                    ix;
                                const size_t wi =
                                    ((static_cast<size_t>(oc) * icg + kic) *
                                         sh.k +
                                     ky) *
                                        sh.k +
                                    kx;
                                sum += static_cast<double>(x[xi]) * w8[wi];
                            }
                        }
                    const size_t yi =
                        ((static_cast<size_t>(n) * sh.oc + oc) * oh + yy) * ow +
                        xx;
                    y[yi] = static_cast<float>(sum) * scale[oc] + bias[oc];
                }
        }
    return y;
}
} // namespace

int main(int argc, char **argv) {
    Logger::getInstance().setLevel(LOG_ERROR);
    if (const char *archive = std::getenv("VKOP_ADA_ARCHIVE")) {
        core::ShaderArchive::set_process_default(archive);
    } else {
        dev::seed_shader_archive_from_build_tree();
    }
    const auto &phydevs =
        VulkanInstance::getVulkanInstance().getPhysicalDevices();
    if (phydevs.empty()) {
        std::printf("no vulkan device\n");
        return 1;
    }
    const char *dev_env = std::getenv("BENCH_DEV");
    const int dev_idx = dev_env ? std::atoi(dev_env) : 0;
    auto dev = std::make_shared<VulkanDevice>(phydevs[dev_idx]);
    printf("GPU: %s\n", dev->getDeviceName().c_str());
    auto cmdpool = std::make_shared<VulkanCommandPool>(dev);

    // OW just below / just above CONV_OW_BLOCK=8 for a 3x3 stride-1 pad-1 conv:
    //   OW = IW + 2p - k + 1 = IW  (p=1, k=3, s=1)
    // so IW=7 -> OW=7 (element), IW=8 -> OW=8 (blocked).
    std::vector<Shape> shapes = {
        {1, 32, 16, 7, 32, 3, 1, 1, 1, "OW=7 element"},
        {1, 32, 16, 8, 32, 3, 1, 1, 1, "OW=8 blocked"},
        {1, 64, 32, 7, 64, 3, 1, 1, 1, "OW=7 element (64ch)"},
        {1, 64, 32, 8, 64, 3, 1, 1, 1, "OW=8 blocked (64ch)"},
    };
    const int reps = (argc > 1) ? std::atoi(argv[1]) : 20;

    for (const auto &sh : shapes) {
        const int oh = (sh.ih + 2 * sh.p - sh.k) / sh.s + 1;
        const int ow = (sh.iw + 2 * sh.p - sh.k) / sh.s + 1;

        std::vector<float> xv(static_cast<size_t>(sh.n) * sh.ic * sh.ih *
                              sh.iw);
        for (size_t i = 0; i < xv.size(); i++) {
            xv[i] = static_cast<float>((i * 37 % 17) - 8) * 0.25f;
        }
        const int icg = sh.ic / sh.g;
        std::vector<int8_t> wv(static_cast<size_t>(sh.oc) * icg * sh.k * sh.k);
        for (size_t i = 0; i < wv.size(); i++) {
            wv[i] = static_cast<int8_t>((i * 53 % 255) - 127);
        }
        std::vector<float> sv(sh.oc), bv(sh.oc);
        for (int i = 0; i < sh.oc; i++) {
            sv[i] = 0.0078125f;
            bv[i] = 0.125f * static_cast<float>(i % 3);
        }

        auto X = std::make_shared<core::Tensor<float>>(
            std::vector<int>{sh.n, sh.ic, sh.ih, sh.iw});
        auto W = std::make_shared<core::Tensor<int8_t>>(
            std::vector<int>{sh.oc, icg, sh.k, sh.k});
        W->set_elem_kind(core::ElemKind::kInt8);
        auto S = std::make_shared<core::Tensor<float>>(std::vector<int>{sh.oc});
        auto B = std::make_shared<core::Tensor<float>>(std::vector<int>{sh.oc});
        auto Y = std::make_shared<core::Tensor<float>>(
            std::vector<int>{sh.n, sh.oc, oh, ow});
        X->fillToCPU(xv);
        W->fillToCPU(wv);
        S->fillToCPU(sv);
        B->fillToCPU(bv);
        X->as_storage_buffer(dev);
        W->as_storage_buffer(dev);
        S->as_storage_buffer(dev);
        B->as_storage_buffer(dev);
        Y->as_storage_buffer(dev);
        X->copyToGPU(cmdpool);
        W->copyToGPU(cmdpool);
        S->copyToGPU(cmdpool);
        B->copyToGPU(cmdpool);

        auto op = ops::create_from_type(ops::OpType::CONV2D, /*fp16=*/0,
                                        /*use_tensorcore=*/0,
                                        /*backend_buffer=*/true);
        op->set_name(sh.name);
        op->set_runtime_device(dev, cmdpool);
        op->enable_replay(true);
        op->setAttribute({{"strides", std::to_string(sh.s)},
                          {"pads", std::to_string(sh.p)},
                          {"dilations", "1"},
                          {"group", std::to_string(sh.g)},
                          {"kernel_shape", std::to_string(sh.k)}});
        std::vector<std::shared_ptr<core::ITensor>> in = {X, W, B, S},
                                                    out = {Y};
        op->onExecute(in, out, 0);
        op->onExecute(in, out, 0);
        auto cmd = op->get_record();

        std::vector<float> want = ref_conv(sh, xv, wv, sv, bv);
        if (cmd) {
            cmd->submit(dev->getComputeQueue(0));
            dev->wait_all_done();
            Y->reserveOnCPU();
            Y->copyToCPU(cmdpool);
            int bad = 0;
            float worst = 0;
            for (size_t i = 0; i < want.size(); i++) {
                float d = std::fabs((*Y)[i] - want[i]);
                worst = std::max(worst, d);
                const float tol = std::max(1e-2f, std::fabs(want[i]) * 1e-3f);
                if (d > tol) {
                    if (bad < 3)
                        std::printf("  mismatch[%zu] got %g want %g\n", i,
                                    (*Y)[i], want[i]);
                    bad++;
                }
            }
            printf("%-18s : %s (worst=%.4g)\n", sh.name, bad ? "WRONG" : "ok",
                   worst);
            if (bad)
                continue;
        } else {
            printf("%-18s : no recorded cmd\n", sh.name);
            continue;
        }

        auto t0 = std::chrono::steady_clock::now();
        for (int r = 0; r < reps; r++)
            cmd->submit(dev->getComputeQueue(0));
        dev->wait_all_done();
        auto t1 = std::chrono::steady_clock::now();
        double ms =
            std::chrono::duration<double, std::milli>(t1 - t0).count() / reps;
        double macs =
            static_cast<double>(sh.n) * sh.oc * oh * ow * sh.ic * sh.k * sh.k;
        printf("%-18s : %8.3f ms  %6.1f GMAC/s\n", sh.name, ms,
               macs / ms / 1e6);
    }
    return 0;
}
