"""Standalone tests for DAGBasedModel._check_unified_region.

The unified region is addressed by a uint32 field in the on-disk UnifiedMeta
(model/vkop_model.fbs), plus an int32 dims field in the same 32-byte struct.
Both have a hard ceiling a big fp16 model can cross, and crossing either one
used to surface as flatbuffers' "bad number <n> for type uint32" from
UnifiedMeta.CreateUnifiedMeta -- at the end of a multi-minute conversion, with
no tensor name and no suggested fix. The check converts that into an
actionable error naming the ceiling, the size, and the way out (drop --unify).

It is checked here rather than as an OptimizationPass because
MultiRoundOptimizer.optimize swallows every exception except
UnknownFusedOpError, so a pass-hosted check would be silently skipped (same
placement reason as the weight-layout audit and _validate_conv_weight_layout).
"""
import pytest

from onnx2vkop.dag import DAGBasedModel, _UNIFIED_MAX_BYTES


def model_with_member(name="w", dims=None, offset=0, size=4, base=0, laid_out=None):
    """A model whose unified region is already laid out, as terminal_liveness_prune
    leaves it on the way into save_to_binary."""
    m = DAGBasedModel()
    m.unified_meta = [{"_name": name, "offset": offset, "size": size,
                       "dims": list(dims if dims is not None else [1]),
                       "name_len": len(name), "dtype": 1}]
    m._unified_bytes = b"\0" * (offset + size if laid_out is None else laid_out)
    m._unified_blob_offset = base
    return m


def test_ordinary_region_passes():
    m = model_with_member()
    m._check_unified_region()  # 4 bytes at the front of an empty blob


def test_no_unified_region_at_all_passes():
    m = DAGBasedModel()
    m.unified_meta = []
    m._check_unified_region()


def test_region_ending_exactly_at_the_ceiling_passes():
    m = model_with_member(offset=_UNIFIED_MAX_BYTES - 4, size=4)
    m._check_unified_region()


def test_region_one_byte_past_the_ceiling_raises():
    m = model_with_member(offset=_UNIFIED_MAX_BYTES - 4, size=5)
    with pytest.raises(ValueError, match="4 GiB"):
        m._check_unified_region()


def test_the_limit_is_the_region_end_not_its_length():
    """The field holds a member's DISTANCE from the region start (the base rides
    in the 64-bit unified_blob_offset), so what has to fit is the region's end,
    not its length."""
    m = model_with_member(offset=_UNIFIED_MAX_BYTES - 1, size=1)
    m._check_unified_region()
    m = model_with_member(offset=_UNIFIED_MAX_BYTES - 1, size=2)
    with pytest.raises(ValueError, match="Drop --unify"):
        m._check_unified_region()


def test_the_base_counts_toward_the_ceiling():
    """The base rides in the 64-bit unified_blob_offset, so the members' offsets
    still fit uint32 at any base -- but the region's END is what the runtime
    sub-allocates, and that is the number the check holds."""
    m = model_with_member(offset=0, size=64, base=_UNIFIED_MAX_BYTES * 3)
    with pytest.raises(ValueError, match="4 GiB"):
        m._check_unified_region()


def test_a_member_with_no_bytes_behind_it_raises():
    """The region is never repacked after Unifier.unify, so a member whose bytes
    were never appended is a region the runtime would sub-allocate past its
    end -- the one failure mode that does not show up at the write path."""
    m = model_with_member(offset=0, size=4, laid_out=0)
    with pytest.raises(ValueError, match="were laid out in the region"):
        m._check_unified_region()


def test_a_repeated_member_is_not_double_counted():
    """Offsets are absolute within the region, not a running total: two members
    at the same offset are two BufferViews over one blob, not 8 bytes."""
    m = DAGBasedModel()
    m.unified_meta = [{"_name": "a", "offset": 0, "size": 4, "dims": [1],
                       "name_len": 1, "dtype": 1},
                      {"_name": "b", "offset": 0, "size": 4, "dims": [1],
                       "name_len": 1, "dtype": 1}]
    m._unified_bytes = b"\0" * 4
    m._unified_blob_offset = 0
    m._check_unified_region()


def test_an_oversized_dim_raises_and_names_the_tensor():
    m = model_with_member(name="lm_head.weight", dims=[1 << 31, 1])
    with pytest.raises(ValueError, match="lm_head.weight"):
        m._check_unified_region()


def test_a_dim_at_the_int32_ceiling_passes():
    m = model_with_member(dims=[0x7FFFFFFF])
    m._check_unified_region()


def test_the_region_message_reports_the_end_the_count_and_the_fix():
    m = model_with_member(offset=0, size=_UNIFIED_MAX_BYTES + 1)
    with pytest.raises(ValueError) as ei:
        m._check_unified_region()
    msg = str(ei.value)
    # It has to say where it stopped and how to get out -- the whole point of
    # the check is that CreateUnifiedMeta's message says neither.
    assert f"{_UNIFIED_MAX_BYTES + 1:,}" in msg
    assert "--unify" in msg
    assert "1 members" in msg
