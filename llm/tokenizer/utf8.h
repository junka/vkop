// UTF-8 流式 sanitizer：把可能跨 token 边界、含非法字节的原始字节流，逐段
// 转成 well-formed UTF-8。移植自 trt_edgellm 的 common/utf8.{h,cpp}（Apache-2.0），
// 仅改 namespace。用于流式 decode：逐 token 拼 piece 字节后过一遍，保证每次
// 产出的增量都是合法 UTF-8，末尾不完整 codepoint 暂存到 pending 下次再处理。
#ifndef LLM_TOKENIZER_UTH8_HPP_
#define LLM_TOKENIZER_UTH8_HPP_
#pragma once

#include <cstdint>
#include <string>

namespace qwen {
namespace utf8 {

// UTF-8 leader 字节的码点长度（1–4）。非合法 leader（孤立 continuation 字节、
// 5+ 字节 leader 等）返回 0。
int leaderByteLen(unsigned char c) noexcept;

// 从 bytes[0..need) 解码一个码点。前置：need==leaderByteLen(bytes[0])>0 且
// 后续 continuation 字节合法。不做 overlong/surrogate/越界校验（见下）。
uint32_t decodeCodepoint(unsigned char const* bytes, int need) noexcept;

// 码点是否能用 need 字节 UTF-8 合法编码（拒绝 overlong、UTF-16 surrogate、
// >U+10FFFF）。need 须在 [1,4]。
bool isValidCodepointForLen(uint32_t cp, int need) noexcept;

// 消费 buffer，产出 well-formed UTF-8。非法字节序列（孤立 continuation、
// overlong、surrogate、越界、伪 leader）替换为 U+FFFD。若 buffer 末尾是不完整
// codepoint（合法 leader 但 continuation 不足），尾部字节移入 pending 留待下次，
// 不输出。pending 是 in-out：旧内容前置拼到 buffer 前，返回时写入新的尾部不完整
// 字节（若有）。输出恒为合法 UTF-8。
std::string sanitizeUtf8Streaming(std::string const& buffer, std::string& pending) noexcept;

// 终止 flush：把 pending 里每个字节各转成一个 U+FFFD 并清空 pending。用于流结束
// 或单次 decode（无后续输入时把残留字节显式暴露成替换字符）。
std::string sanitizeUtf8Flush(std::string& pending) noexcept;

} // namespace utf8
} // namespace qwen
#endif /* LLM_TOKENIZER_UTH8_HPP_ */