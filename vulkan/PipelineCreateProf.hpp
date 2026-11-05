// Copyright 2026 @junka
#ifndef VULKAN_PIPELINECREATEPROF_HPP_
#define VULKAN_PIPELINECREATEPROF_HPP_

#include <cstdint>

namespace vkop {

// Timing probe for vkCreateComputePipelines, gated by
// VK_PIPELINE_CREATE_PROF=1.
//
// It answers one question: how much of a model load (and of the first dispatch
// of a new shape) goes through vkCreateComputePipelines, and whether a
// persistent pipeline cache removes that cost. The report splits creates at a
// duration threshold, which is a display hint only: measured deserialization of
// a large cached SPIR-V sits in the same range as a short compile, so the split
// does not prove compilation happened. VK_EXT_pipeline_creation_feedback is the
// extension that reports a real cache hit per pipeline; it is not wired in here.
//
// Disabled cost is one cached-bool check per call: pipeline_prof_now() returns
// 0 and the setters return before touching state.
bool pipeline_prof_enabled();

// Monotonic ns stamp, 0 when disabled. Only the delta of two calls while
// enabled is meaningful.
uint64_t pipeline_prof_now();

// "load" while Runtime::LoadModel is running, "run" otherwise. Ops build
// secondary pipelines lazily on the first dispatch of a new shape, so the
// phase is what separates a cold-load compile from a mid-session one.
void pipeline_prof_set_phase(const char *phase);

// Graph node whose op is currently building, for slowest-entry attribution.
// Copied, so call it only from the load path; empty it again when the phase
// changes.
void pipeline_prof_set_context(const char *context);

// create_ns is the vkCreateComputePipelines call itself, ctor_ns the whole
// VulkanPipeline construction (shader module, descriptor layout, pipeline
// layout, pool). Records nothing when disabled.
void pipeline_prof_record(uint64_t create_ns, uint64_t ctor_ns,
                          uint32_t spv_bytes, uint32_t required_subgroup_size);

// One block per phase seen so far plus a total. span_ms, when nonzero, is the
// enclosing operation (a model load), and the report shows what fraction of it
// was pipeline creation. Repeatable; nothing resets.
void pipeline_prof_report(const char *header, double span_ms = 0.0);

} // namespace vkop

#endif // VULKAN_PIPELINECREATEPROF_HPP_
