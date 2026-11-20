#!/usr/bin/env bash
# Build and run the full round trip on the PPU host over SSH.
#
# The PPU host has libvulkan.so.1 but no Vulkan headers, no SPIRV-Tools
# development package, and no glslc, so all three are copied from the machine
# running this script and vendored into the remote build tree. The PPU SDK
# itself (llc, nvcc, libcuda.so.1) is on the host.
#
# Usage: scripts/run-on-gong.sh [host] [device-index]
set -euo pipefail

host="${1:-gong}"
device_index="${2:-1}"  # device 0 is busy with other people's work on the shared box
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
remote="/tmp/vkppu-icd"

spirv_tools_inc="${SPIRV_TOOLS_INCLUDE_DIR:-/usr/include}"
spirv_tools_lib="${SPIRV_TOOLS_LIBRARY:-/usr/lib/x86_64-linux-gnu/libSPIRV-Tools.a}"

for f in /usr/include/vulkan/vulkan_core.h "${spirv_tools_inc}/spirv-tools/libspirv.h" \
         "${spirv_tools_lib}"; do
  if [ ! -e "$f" ]; then
    echo "error: missing local dependency: $f" >&2
    exit 1
  fi
done
if ! command -v glslc >/dev/null 2>&1; then
  echo "error: glslc not found; it is used to compile the test shader locally" >&2
  exit 1
fi

echo "== copying sources and headers to ${host}:${remote} =="
ssh "${host}" "rm -rf '${remote}' && mkdir -p '${remote}/third_party'"
scp -q -r "${here}/src" "${here}/tests" "${here}/spirv-emit" "${here}/CMakeLists.txt" \
  "${here}/ppu_icd.json.in" "${host}:${remote}/"
scp -q -r /usr/include/vulkan "${host}:${remote}/third_party/"
# vulkan_core.h includes vk_video/... which lives beside vulkan/ on Debian.
if [ -d /usr/include/vk_video ]; then
  scp -q -r /usr/include/vk_video "${host}:${remote}/third_party/"
fi
# SPIRV-Tools is what the in-process emitter parses with. It is static-only on
# Debian/Ubuntu, so the archive travels with the tree.
# Layout under third_party/: spirv-tools/libspirv.h (the tools API),
# spirv/unified1/spirv.h (the enum/opcode headers) and libSPIRV-Tools.a.
scp -q -r "${spirv_tools_inc}/spirv-tools" "${spirv_tools_inc}/spirv" "${host}:${remote}/third_party/"
scp -q "${spirv_tools_lib}" "${host}:${remote}/third_party/libSPIRV-Tools.a"

echo "== building on ${host} =="
ssh "${host}" "bash -lc '
  set -euo pipefail
  cd ${remote}
  cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DPPU_SPIRV_TOOLS_INCLUDE_DIR=${remote}/third_party \
    -DPPU_SPIRV_TOOLS_LIBRARY=${remote}/third_party/libSPIRV-Tools.a 2>&1 \
    | grep -E \"SPIRV-Tools|error\" || true
  cmake --build build -j\$(nproc) 2>&1 | tail -3
'"

echo "== writing manifest =="
ssh "${host}" "cat > ${remote}/ppu.json <<'JSON'
{
  \"file_format_version\": \"1.0.0\",
  \"ICD\": { \"api_version\": \"1.1.0\", \"library_path\": \"${remote}/build/libvk_ppu_icd.so\" }
}
JSON"

echo "== building a test kernel (cubin) =="
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

echo "== compiling the SPIR-V test shader locally =="
# --target-env=vulkan1.1 matches what vkop's shader build produces (SPIR-V 1.1),
# which is the encoding the ICD sees in practice.
tmp_spv="$(mktemp -d)"
trap 'rm -rf "${tmp_spv}"' EXIT
glslc --target-env=vulkan1.1 -fshader-stage=compute \
  "${here}/tests/shaders/simple_mul.comp" -o "${tmp_spv}/simple_mul.spv"
scp -q "${tmp_spv}/simple_mul.spv" "${host}:${remote}/simple_mul.spv"

echo "== round trip (cubin path) on device ${device_index} =="
ssh "${host}" "bash -lc '
  source /usr/local/PPU_SDK/envsetup.sh >/dev/null 2>&1
  cd ${remote}
  VK_ICD_FILENAMES=${remote}/ppu.json VK_PPU_DEVICE_INDEX=${device_index} \
    ./build/ppu_roundtrip vecadd.cubin
'"

echo "== SPIR-V compile + dispatch on device ${device_index} =="
ssh "${host}" "bash -lc '
  source /usr/local/PPU_SDK/envsetup.sh >/dev/null 2>&1
  cd ${remote}
  VK_ICD_FILENAMES=${remote}/ppu.json VK_PPU_DEVICE_INDEX=${device_index} \
    ./build/ppu_spirv_dispatch ${remote}/simple_mul.spv
'"
