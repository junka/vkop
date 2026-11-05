// Copyright 2026 @junka
#ifndef VULKAN_DESCRIPTORWRITEPROF_HPP_
#define VULKAN_DESCRIPTORWRITEPROF_HPP_

#include <cstdint>

namespace vkop {

// Timing probe for the host-side descriptor management calls
// (vkUpdateDescriptorSets, vkAllocateDescriptorSets, vkFreeDescriptorSets),
// gated by VK_DS_WRITE_PROF=1.
//
// It answers the ceiling question for VK_KHR_push_descriptor: that extension
// removes vkUpdateDescriptorSets and the descriptor-set allocation from the
// dispatch record path, so the summed time of those calls is the most a
// push-descriptor rewrite could save. It is a ceiling, not an estimate of the
// win — a pushed descriptor still costs a vkCmdPushDescriptorSetKHR in the
// command buffer, and ops that rely on distinct descriptor sets per pass
// (submit_per_ds) or on replay reusing a recording need reworking.
//
// Disabled cost is one cached-bool check per call.
bool ds_prof_enabled();

// Monotonic ns stamp, 0 when disabled. Only the delta of two calls while
// enabled is meaningful.
uint64_t ds_prof_now();

// "load" while Runtime::LoadModel is running, "run" otherwise. Same meaning as
// the pipeline-create probe's phase.
void ds_prof_set_phase(const char *phase);

// kind is a static string: "update", "alloc", "free", "bind_pipeline",
// "bind_ds" or "push_const". count is the number of VkWriteDescriptorSet
// entries for "update", otherwise ignored.
void ds_prof_record(const char *kind, uint64_t ns, uint32_t count);

// One block per phase/kind seen so far plus totals. span_ms, when nonzero, is
// the enclosing operation and the report shows what fraction of it descriptor
// management took. Both report functions PRINT AND THEN RESET, so a report
// covers only the calls since the previous one; nothing accumulates across them.
void ds_prof_report(const char *header, double span_ms = 0.0);

// One line per dispatch round (Runtime::Run), for the time series: the two
// columns VK_KHR_push_descriptor acts on (update, bind_ds) plus the columns it
// keeps, and the round's total. Resets like ds_prof_report().
void ds_prof_round(double span_ms);

} // namespace vkop

#endif // VULKAN_DESCRIPTORWRITEPROF_HPP_
