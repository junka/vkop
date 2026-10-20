// Kernel-plan reader test (Model.kernel_plans / Node.plan).
//
// onnx2vkop writes the kernel selection it used into the .vkopbin, and the
// runtime looks every op up by it (Runtime::kernel_key_for_node). That only
// helps if the table says exactly what the C++ side would otherwise have
// derived, so this loads a real converted fixture and checks three things:
//
//   1. every row is what core/ShaderKeys.hpp::shader_key_for() returns for all
//      four (backend, precision) combinations -- i.e. the file's answer and
//      the built-in table agree for the ops it plans;
//   2. every key a plan lists (cells and variants) is inside the model's own
//      embedded archive. That is the invariant that makes
//      VKOP_STRIP_BUILTIN_SHADERS=ON safe: with no linked-in SPIR-V left, a
//      key that is planned but not embedded is a hard failure at pipeline
//      creation rather than a fallback;
//   3. a node without a plan has no key at all under the built-in table, so
//      Node.plan = -1 can only ever mean "nothing to look up".
//
// Plain main (no gtest, no torch) so it stays cheap to rebuild and runs in the
// strip build too. Fixture comes from argv[1] (the CMake-converted artifact).

#include <cstdio>
#include <string>
#include <vector>

#include "core/KernelPlan.hpp"
#include "core/ShaderArchive.hpp"
#include "core/ShaderKeys.hpp"
#include "model/load.hpp"
#include "ops/Ops.hpp"

namespace {

int failures = 0;

void fail(const std::string &msg) {
    std::fprintf(stderr, "FAIL: %s\n", msg.c_str());
    ++failures;
}

} // namespace

int main(int argc, char **argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <model.vkopbin>\n", argv[0]);
        return 2;
    }
    const std::string path = argv[1];

    std::unique_ptr<vkop::load::VkModel> model;
    try {
        model = std::make_unique<vkop::load::VkModel>(path);
    } catch (const std::exception &e) {
        std::fprintf(stderr, "FAIL: cannot load %s: %s\n", path.c_str(), e.what());
        return 2;
    }

    const auto &plans = model->kernel_plans;
    std::printf("kernel plans: %zu row(s) for %zu node(s)\n", plans.size(),
                model->nodes.size());
    if (plans.empty()) {
        std::fprintf(stderr,
                     "FAIL: %s carries no kernel plan; it was converted by an "
                     "older writer or the field was dropped\n",
                     path.c_str());
        return 2;
    }

    // Rows are deduplicated by op_type; a duplicate means the writer's index
    // building is broken (and Node.plan would be ambiguous).
    for (size_t i = 0; i < plans.size(); ++i) {
        if (plans[i].empty() || plans[i].op_type.empty()) {
            fail("plan row " + std::to_string(i) + " is empty");
        }
        for (size_t j = i + 1; j < plans.size(); ++j) {
            if (plans[i].op_type == plans[j].op_type) {
                fail("duplicate plan row for op_type " + plans[i].op_type);
            }
        }
    }

    // 1 + 3: the plan and the built-in table must agree, and -1 must mean
    // "no key".
    int planned_nodes = 0;
    int checked_keys = 0;
    for (const auto &node : model->nodes) {
        const auto type = vkop::ops::convert_opstring_to_enum(node.op_type);
        if (node.plan_index < 0) {
            for (int backend_buffer : {0, 1}) {
                for (int fp16 : {0, 1}) {
                    const auto k = vkop::core::shader_key_for(
                        type, backend_buffer != 0, fp16);
                    if (!k.empty()) {
                        fail("node " + node.name + " (" + node.op_type +
                             ") has no plan but the built-in table yields key " +
                             k);
                    }
                }
            }
            continue;
        }
        if (node.plan_index >= static_cast<int>(plans.size())) {
            fail("node " + node.name + " has out-of-range plan=" +
                 std::to_string(node.plan_index));
            continue;
        }
        ++planned_nodes;
        const auto &plan = plans[node.plan_index];
        if (plan.op_type != node.op_type) {
            fail("node " + node.name + " (" + node.op_type +
                 ") points at a plan for " + plan.op_type);
            continue;
        }
        for (int backend_buffer : {0, 1}) {
            for (int fp16 : {0, 1}) {
                const auto want = vkop::core::shader_key_for(
                    type, backend_buffer != 0, fp16);
                const auto &got = plan.key(backend_buffer != 0, fp16);
                if (got != want) {
                    fail("plan(" + plan.op_type + ") backend=" +
                         (backend_buffer ? "buffer" : "image") + " fp16=" +
                         std::to_string(fp16) + " says '" + got +
                         "', the built-in table says '" + want + "'");
                }
                ++checked_keys;
            }
        }
    }
    if (planned_nodes == 0) {
        fail("no node in " + path + " carries a plan");
    }

    // 2: the plan must be covered by the archive the file itself carries.
    if (model->shader_archive_size == 0) {
        fail(path + " has a kernel plan but no embedded shader archive");
    } else {
        auto archive = vkop::core::ShaderArchive::open_from_memory(
            model->shader_archive_memory, model->shader_archive_size);
        if (!archive) {
            fail(path + "'s embedded archive does not parse");
        } else {
            for (const auto &plan : plans) {
                std::vector<std::string> keys{plan.image, plan.image_fp16,
                                              plan.buffer, plan.buffer_fp16};
                keys.insert(keys.end(), plan.variants.begin(),
                            plan.variants.end());
                for (const auto &k : keys) {
                    if (k.empty()) {
                        continue;
                    }
                    if (!archive->lookup(k).first) {
                        fail("plan(" + plan.op_type +
                             ") lists key '" + k +
                             "' which the model's own archive does not carry");
                    }
                    ++checked_keys;
                }
            }
        }
    }

    std::printf("kernel plans: %d planned node(s), %d key(s) checked against "
                "the built-in table and the embedded archive, %d failure(s)\n",
                planned_nodes, checked_keys, failures);
    return failures == 0 ? 0 : 1;
}
