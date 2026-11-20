#!/usr/bin/env bash
# Build the ICD locally and run the loader-handshake checks.
#
# On a host with no accelerator this uses the opt-in synthetic device, which
# exercises enumeration and every physical-device query. The real round trip
# needs an actual PPU; see run-on-gong.sh.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build="${here}/build"

cmake -S "${here}" -B "${build}" -DCMAKE_BUILD_TYPE=RelWithDebInfo >/dev/null
cmake --build "${build}" -j"$(nproc)"

# CMake writes ppu_icd.json pointing at the built library.
manifest="${build}/ppu_icd.json"
echo "manifest: ${manifest}"

echo
echo "== vulkaninfo --summary =="
if command -v vulkaninfo >/dev/null 2>&1; then
  # VK_PPU_SYNTHETIC_DEVICE only matters when no accelerator is present.
  VK_ICD_FILENAMES="${manifest}" VK_PPU_SYNTHETIC_DEVICE=1 \
    vulkaninfo --summary 2>&1 | grep -v 'GetInstanceProcAddr' | sed -n '/^Devices:/,$p'
else
  echo "vulkaninfo not installed; skipping"
fi

echo
echo "== roundtrip (phase A) =="
VK_ICD_FILENAMES="${manifest}" VK_PPU_SYNTHETIC_DEVICE=1 "${build}/ppu_roundtrip"
