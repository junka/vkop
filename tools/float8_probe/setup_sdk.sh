#!/usr/bin/env bash
# Install a second, newer Vulkan SDK side by side and build the float8 probe
# against it. Nothing in this script touches the system loader or the
# existing 1.3.275 headers: the SDK lands under $SDK_ROOT and only the
# commands this script launches see it (through LD_LIBRARY_PATH and an
# explicit -I/-L).
#
# Why this exists: VK_EXT_shader_float8 (and the GL_EXT_float_e4m3 /
# GL_EXT_float_e5m2 GLSL types, plus Float8CooperativeMatrixEXT) are not in
# the 1.3.275 headers, loader or glslc that both the dev box and b3 ship.
# The *driver* already exposes them -- NVIDIA 595 on Ada reports
# shaderFloat8 = true and lists a 16x16x32 E4M3 cooperative-matrix config --
# so the only missing pieces are headers, loader and shader compiler.
#
# Usage:
#   tools/float8_probe/setup_sdk.sh [SDK_VERSION] [SDK_ROOT]
#
#   SDK_VERSION  defaults to 1.4.363.1 (latest at time of writing). Any version
#                in https://vulkan.lunarg.com/sdk/versions.json >= 1.4.321.0
#                works; 1.4.317 is a Vulkan-Headers tag, NOT an SDK release.
#   SDK_ROOT     defaults to $HOME/vulkan-sdk
#
# Environment knobs:
#   SKIP_DOWNLOAD=1   reuse an already extracted $SDK_ROOT/$SDK_VERSION
#   F8_BUILD_DIR      where the probe and SPIR-V go (default /tmp/f8build)
set -euo pipefail

SDK_VERSION="${1:-1.4.363.1}"
SDK_ROOT="${2:-$HOME/vulkan-sdk}"
F8_BUILD_DIR="${F8_BUILD_DIR:-/tmp/f8build}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

arch="$(uname -m)"
case "$arch" in
x86_64) sdk_url="https://sdk.lunarg.com/sdk/download/${SDK_VERSION}/linux/vulkansdk-linux-x86_64-${SDK_VERSION}.tar.xz" ;;
aarch64) sdk_url="https://sdk.lunarg.com/sdk/download/${SDK_VERSION}/linux/vulkansdk-linux-aarch64-${SDK_VERSION}.tar.xz" ;;
*) echo "unsupported arch $arch" >&2; exit 1 ;;
esac

if [ "${SKIP_DOWNLOAD:-0}" != "1" ] && [ ! -d "$SDK_ROOT/$SDK_VERSION" ]; then
    mkdir -p "$SDK_ROOT"
    if [ ! -f "$SDK_ROOT/vulkansdk-${arch}-${SDK_VERSION}.tar.xz" ]; then
        echo "==> downloading Vulkan SDK $SDK_VERSION (~350 MB)"
        curl -fL --retry 3 -o "$SDK_ROOT/vulkansdk-${arch}-${SDK_VERSION}.tar.xz" "$sdk_url"
    fi
    echo "==> extracting"
    tar -xf "$SDK_ROOT/vulkansdk-${arch}-${SDK_VERSION}.tar.xz" -C "$SDK_ROOT"
fi

SDK="$SDK_ROOT/$SDK_VERSION/$arch"
if [ ! -d "$SDK" ]; then
    echo "expected SDK at $SDK -- check the version string" >&2
    exit 1
fi
export LD_LIBRARY_PATH="$SDK/lib:$SDK/lib/VulkanLoader/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

hdr_ver="$(sed -n 's/^#define VK_HEADER_VERSION \([0-9]*\)$/\1/p' "$SDK/include/vulkan/vulkan_core.h")"
echo "==> headers: $SDK/include/vulkan (VK_HEADER_VERSION=$hdr_ver)"
echo "==> loader : $SDK/lib/VulkanLoader/lib/libvulkan.so.1.${SDK_VERSION%.*}"
echo "==> glslc  : $SDK/bin/glslc"

if [ "${hdr_ver:-0}" -lt 317 ]; then
    echo "headers are older than the float8 extension (need >= 317)" >&2
    exit 1
fi

mkdir -p "$F8_BUILD_DIR"

# The repo's own shaders are compiled with plain --target-spv=spv1.3 and the
# default target env; float8 also compiles under those flags. vulkan1.3 is
# used here because it is the lowest env that accepts the fp8 GLSL types
# without the memory-model extension warning.
echo "==> compiling fp8 shaders"
"$SDK/bin/glslc" -c "$HERE/f8_storage.comp" -fshader-stage=compute \
    --target-env=vulkan1.3 -o "$F8_BUILD_DIR/f8_storage.spv"
"$SDK/bin/glslc" -c "$HERE/f8_coopmat.comp" -fshader-stage=compute \
    --target-env=vulkan1.3 -o "$F8_BUILD_DIR/f8_coopmat.spv"

# Sanity: the SPIR-V must actually carry the float8 capabilities. Without this
# a silently-ignored extension would still produce a plain shader that the
# device accepts, and the probe would "pass" without testing anything.
for f in f8_storage f8_coopmat; do
    if ! "$SDK/bin/spirv-dis" "$F8_BUILD_DIR/$f.spv" | grep -q 'OpCapability Float8EXT'; then
        echo "$f.spv is missing OpCapability Float8EXT -- glslc ignored the extension" >&2
        exit 1
    fi
done
if ! "$SDK/bin/spirv-dis" "$F8_BUILD_DIR/f8_coopmat.spv" | grep -q 'Float8CooperativeMatrixEXT'; then
    echo "f8_coopmat.spv is missing Float8CooperativeMatrixEXT" >&2
    exit 1
fi
echo "    both SPIR-V modules carry the float8 capabilities"

echo "==> building the probe"
CXX="${CXX:-g++}"
"$CXX" -std=c++17 -O2 -o "$F8_BUILD_DIR/f8_probe" "$HERE/f8_probe.cpp" \
    -I"$SDK/include" -L"$SDK/lib/VulkanLoader/lib" -lvulkan

cat <<EOF

==> done.

Run it with the SDK loader in front of the system one:

  LD_LIBRARY_PATH=$SDK/lib/VulkanLoader/lib \\
      $F8_BUILD_DIR/f8_probe $F8_BUILD_DIR/f8_storage.spv $F8_BUILD_DIR/f8_coopmat.spv

Exit 0 with "OVERALL PASS" means both the storage round trip and the fp8
cooperative-matrix MMA ran and matched. "SKIP: VK_EXT_shader_float8 not
advertised" is a normal result on a device without the extension.
EOF
