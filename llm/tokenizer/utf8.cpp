// UTF-8 流式 sanitizer 实现。移植自 trt_edgellm common/utf8.cpp（Apache-2.0），
// 仅改 namespace 与 include 路径。算法说明见 utf8.h。
#include "utf8.h"

#include <cstdint>

namespace qwen {
namespace utf8 {

namespace {
constexpr char kFFFD[] = "\xEF\xBF\xBD"; // U+FFFD REPLACEMENT CHARACTER 的 UTF-8
constexpr int kFFFDLen = 3;
} // namespace

int leaderByteLen(unsigned char c) noexcept
{
    if ((c & 0x80) == 0x00) return 1; // ASCII
    if ((c & 0xE0) == 0xC0) return 2;
    if ((c & 0xF0) == 0xE0) return 3;
    if ((c & 0xF8) == 0xF0) return 4;
    return 0; // 非法 leader / 孤立 continuation
}

uint32_t decodeCodepoint(unsigned char const* bytes, int need) noexcept
{
    switch (need) {
    case 1: return static_cast<uint32_t>(bytes[0]);
    case 2: return (static_cast<uint32_t>(bytes[0] & 0x1F) << 6) | static_cast<uint32_t>(bytes[1] & 0x3F);
    case 3:
        return (static_cast<uint32_t>(bytes[0] & 0x0F) << 12) | (static_cast<uint32_t>(bytes[1] & 0x3F) << 6)
            | static_cast<uint32_t>(bytes[2] & 0x3F);
    case 4:
        return (static_cast<uint32_t>(bytes[0] & 0x07) << 18) | (static_cast<uint32_t>(bytes[1] & 0x3F) << 12)
            | (static_cast<uint32_t>(bytes[2] & 0x3F) << 6) | static_cast<uint32_t>(bytes[3] & 0x3F);
    default: return 0; // 前置被违反，不应到达
    }
}

bool isValidCodepointForLen(uint32_t cp, int need) noexcept
{
    bool const overlong = (need == 2 && cp < 0x80) || (need == 3 && cp < 0x800) || (need == 4 && cp < 0x10000);
    bool const surrogate = (cp >= 0xD800 && cp <= 0xDFFF);
    bool const tooBig = (cp > 0x10FFFF);
    return !overlong && !surrogate && !tooBig;
}

std::string sanitizeUtf8Streaming(std::string const& buffer, std::string& pending) noexcept
{
    // 先把上次残留的不完整字节前置拼到 buffer 前，并清空 pending。
    std::string input = std::move(pending);
    input.append(buffer);
    pending.clear();

    std::string out;
    out.reserve(input.size());

    auto const* bytes = reinterpret_cast<unsigned char const*>(input.data());
    size_t i = 0;
    while (i < input.size()) {
        unsigned char const c = bytes[i];
        int const need = leaderByteLen(c);
        if (need == 0) {
            // 非法 leader 或孤立 continuation：替换，前进 1 字节。
            out.append(kFFFD, kFFFDLen);
            i += 1;
            continue;
        }
        if (i + static_cast<size_t>(need) > input.size()) {
            // 末尾不完整 codepoint：暂存到 pending，等下次。
            pending.assign(input, i, input.size() - i);
            break;
        }
        // 校验 continuation 字节。
        bool valid = true;
        for (int k = 1; k < need; ++k) {
            if ((bytes[i + static_cast<size_t>(k)] & 0xC0) != 0x80) {
                valid = false;
                break;
            }
        }
        if (!valid) {
            out.append(kFFFD, kFFFDLen);
            i += 1;
            continue;
        }
        uint32_t const cp = decodeCodepoint(bytes + i, need);
        if (!isValidCodepointForLen(cp, need)) {
            out.append(kFFFD, kFFFDLen);
            i += 1;
            continue;
        }
        out.append(input, i, static_cast<size_t>(need));
        i += static_cast<size_t>(need);
    }
    return out;
}

std::string sanitizeUtf8Flush(std::string& pending) noexcept
{
    std::string out;
    out.reserve(pending.size() * kFFFDLen);
    for (size_t k = 0; k < pending.size(); ++k) {
        out.append(kFFFD, kFFFDLen);
    }
    pending.clear();
    return out;
}

} // namespace utf8
} // namespace qwen
