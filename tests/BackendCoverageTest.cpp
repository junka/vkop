// Copyright 2026 @junka
// Report-only ledger of which backend each op can actually serve, so
// double-backend test coverage can be prioritised instead of guessed at.
//
// The façade picks the buffer impl only when the op ships a buffer port, and
// silently keeps the image impl otherwise — so asking create_from_type +
// uses_buffer_backend() is the only source of truth that cannot drift from
// the shader tree.
//
// fp16 is a separate axis and is NOT reported here: a leaf may pass its fp32
// spv even when fp16 is requested (see the fp16 note in
// ops/BufferUnaryFactory.hpp), so presence of an fp16 build has to be read
// off the shader build, not off the op class.

#include <exception>
#include <string>
#include <vector>

#include "setup.hpp"
#include "include/logger.hpp"
#include "ops/OperatorFactory.hpp"
#include "ops/Ops.hpp"

#include <gtest/gtest.h>

namespace {

struct Row {
    std::string name;
    bool image;
    bool buffer;
};

bool serves(vkop::ops::OpType type, bool want_buffer) {
    // Buffer-only façades (RotaryEmbedding, Where, ...) throw when asked for
    // the image impl, so a throw means "no such backend", not "test failed".
    try {
        auto op = vkop::ops::create_from_type(type, 0, 0, want_buffer);
        return op && op->uses_buffer_backend() == want_buffer;
    } catch (const std::exception &) {
        return false;
    }
}

std::vector<Row> collect_capability_rows() {
    std::vector<Row> rows;
    for (int i = static_cast<int>(vkop::ops::OpType::UNKNOWN) + 1;
         i < static_cast<int>(vkop::ops::OpType::TOTAL_NUM); ++i) {
        const auto type = static_cast<vkop::ops::OpType>(i);
        rows.push_back({vkop::ops::convert_optype_to_string(type),
                        serves(type, /*want_buffer=*/false),
                        serves(type, /*want_buffer=*/true)});
    }
    return rows;
}

}  // namespace

TEST(BackendCoverageTest, CapabilityTable) {
    const auto rows = collect_capability_rows();
    int both = 0, image_only = 0, buffer_only = 0, neither = 0;
    LOG_INFO("=== op backend capability (image = image2DArray, "
             "buffer = SSBO; fp32 probe) ===");
    for (const auto &r : rows) {
        const char *verdict = r.image && r.buffer        ? "both"
                              : r.image                  ? "image-only"
                              : r.buffer                 ? "buffer-only"
                                                         : "NEITHER";
        LOG_INFO("%-18s %-11s (image=%d buffer=%d)", r.name.c_str(), verdict,
                 r.image, r.buffer);
        if (r.image && r.buffer) both++;
        else if (r.image) image_only++;
        else if (r.buffer) buffer_only++;
        else neither++;
    }
    LOG_INFO("=== totals: %zu ops, both=%d image-only=%d buffer-only=%d "
             "neither=%d ===",
             rows.size(), both, image_only, buffer_only, neither);

    // The table is only useful if it is not trivially empty or all-one-backend;
    // these guard against the probe itself breaking (e.g. a factory that stops
    // honouring the flag), which would silently turn every dual run into a
    // duplicate of the same shader.
    EXPECT_EQ(rows.size(),
              static_cast<size_t>(vkop::ops::OpType::TOTAL_NUM) - 1);
    EXPECT_GT(both, 0);
    EXPECT_GT(buffer_only, 0);
    EXPECT_EQ(neither, 0) << "an op served by no backend means the probe lost "
                            "the flag, not that coverage is missing";
}
