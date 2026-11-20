#!/usr/bin/env bash
# Build and run the full round trip on the PPU host over SSH.
#
# The PPU host has libvulkan.so.1 but no Vulkan headers and no dev symlink, so
# the headers are copied from the machine running this script and vendored into
# the remote build tree.
#
# Usage: scripts/run-on-gong.sh [host] [device-index]
set -euo pipefail

host="${1:-gong}"
device_index="${2:-1}"  # device 0 is busy with other people's work on the shared box
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
remote="/tmp/vkppu-icd"

if [ ! -f /usr/include/vulkan/vulkan_core.h ]; then
  echo "error: Vulkan headers not found locally (looked for /usr/include/vulkan/vulkan_core.h)" >&2
  exit 1
fi

echo "== copying sources and headers to ${host}:${remote} =="
ssh "${host}" "rm -rf '${remote}' && mkdir -p '${remote}/third_party'"
scp -q -r "${here}/src" "${here}/tests" "${here}/CMakeLists.txt" "${here}/ppu_icd.json.in" \
  "${host}:${remote}/"
scp -q -r /usr/include/vulkan "${host}:${remote}/third_party/"
# vulkan_core.h includes vk_video/... which lives beside vulkan/ on Debian.
if [ -d /usr/include/vk_video ]; then
  scp -q -r /usr/include/vk_video "${host}:${remote}/third_party/"
fi

echo "== building on ${host} =="
ssh "${host}" "bash -lc '
  set -euo pipefail
  cd ${remote}
  cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo >/dev/null
  cmake --build build -j\$(nproc) 2>&1 | tail -2
'"

echo "== writing manifest =="
ssh "${host}" "cat > ${remote}/ppu.json <<'JSON'
{
  \"file_format_version\": \"1.0.0\",
  \"ICD\": { \"api_version\": \"1.1.0\", \"library_path\": \"${remote}/build/libvk_ppu_icd.so\" }
}
JSON"

echo "== building a test kernel =="
ssh "${host}" "bash -lc '
  source /usr/local/PPU_SDK/envsetup.sh >/dev/null 2>&1
  cd ${remote}
  cat > vecadd.cu <<\"CU\"
extern \"C\" __global__ void vecadd(const float* a, const float* b, float* c) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < 1024) c[i] = a[i] + b[i];
}
CU
  nvcc -arch=sm_80 -cubin -o vecadd.cubin vecadd.cu 2>/dev/null
'"

echo "== running round trip on device ${device_index} =="
ssh "${host}" "bash -lc '
  source /usr/local/PPU_SDK/envsetup.sh >/dev/null 2>&1
  cd ${remote}
  VK_ICD_FILENAMES=${remote}/ppu.json VK_PPU_DEVICE_INDEX=${device_index} \
    ./build/ppu_roundtrip vecadd.cubin
'"
