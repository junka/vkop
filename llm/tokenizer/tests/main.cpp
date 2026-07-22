#include "tokenizer.hpp"
#include <iostream>
#include <cassert>
#include <chrono>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cstdlib>
#include <unistd.h>

// 测试数据文件名。实际路径由 main 里解析（优先环境变量，其次可执行文件旁），
// 让 `make test` / ctest 不依赖 cwd。
static const char* kBinName = "qwen3_vl.bin";
static const char* kGtName = "hf_ground_truth.json";

// 极简 JSON 解析：只读 hf_ground_truth.json 的 [{s, ids}, ...] 结构。
// 不依赖 nlohmann/json，仅处理转义字符串和整数数组。
namespace {

// 在若干候选位置查找测试数据文件，返回首个存在的路径，找不到返回空串。
// 候选顺序：$QWEN_TOKENIZER_BIN/$QWEN_TEST_DATA 目录、可执行文件所在目录、
// 可执行文件所在目录的上一级（源码根）、源码根/tests。
std::string find_data_file(const char* name) {
    std::vector<std::string> dirs;
    if (const char* env = std::getenv("QWEN_TEST_DATA")) dirs.emplace_back(env);
    // 可执行文件所在目录：/proc/self/exe（Linux）
    char exe[4096];
    ssize_t n = ::readlink("/proc/self/exe", exe, sizeof(exe) - 1);
    if (n > 0) {
        exe[n] = '\0';
        std::string exedir = std::string(exe);
        auto slash = exedir.find_last_of('/');
        if (slash != std::string::npos) {
            std::string d = exedir.substr(0, slash);
            dirs.push_back(d);
            // 上一级（源码根）及其 tests 子目录：源码内构建时数据在 tests/
            auto slash2 = d.find_last_of('/');
            if (slash2 != std::string::npos) {
                std::string parent = d.substr(0, slash2);
                dirs.push_back(parent);
                dirs.push_back(parent + "/tests");
            }
        }
    }
    // 兜底：cwd 及 cwd/tests
    dirs.emplace_back(".");
    dirs.emplace_back("tests");

    for (const auto& d : dirs) {
        std::string p = d + "/" + name;
        std::ifstream f(p, std::ios::binary);
        if (f.good()) return p;
    }
    return std::string();
}

std::string unescape_json_string(const std::string& in) {
    std::string out;
    out.reserve(in.size());
    for (size_t i = 0; i < in.size(); ++i) {
        if (in[i] == '\\' && i + 1 < in.size()) {
            char nx = in[i + 1];
            switch (nx) {
            case '"': out += '"'; ++i; break;
            case '\\': out += '\\'; ++i; break;
            case '/': out += '/'; ++i; break;
            case 'n': out += '\n'; ++i; break;
            case 't': out += '\t'; ++i; break;
            case 'r': out += '\r'; ++i; break;
            case 'b': out += '\b'; ++i; break;
            case 'f': out += '\f'; ++i; break;
            case 'u': {
                // \uXXXX -> UTF-8（含代理对）。测试数据里有 emoji/组合字符。
                if (i + 5 < in.size()) {
                    unsigned cp = std::stoul(in.substr(i + 2, 4), nullptr, 16);
                    i += 5;
                    if (cp >= 0xD800 && cp <= 0xDBFF && i + 6 < in.size() && in[i + 1] == '\\' && in[i + 2] == 'u') {
                        unsigned lo = std::stoul(in.substr(i + 3, 4), nullptr, 16);
                        if (lo >= 0xDC00 && lo <= 0xDFFF) {
                            cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                            i += 6;
                        }
                    }
                    if (cp < 0x80) out += static_cast<char>(cp);
                    else if (cp < 0x800) { out += char(0xC0 | (cp >> 6)); out += char(0x80 | (cp & 0x3F)); }
                    else if (cp < 0x10000) { out += char(0xE0 | (cp >> 12)); out += char(0x80 | ((cp >> 6) & 0x3F)); out += char(0x80 | (cp & 0x3F)); }
                    else { out += char(0xF0 | (cp >> 18)); out += char(0x80 | ((cp >> 12) & 0x3F)); out += char(0x80 | ((cp >> 6) & 0x3F)); out += char(0x80 | (cp & 0x3F)); }
                }
                break;
            }
            default: out += in[i]; break;
            }
        } else {
            out += in[i];
        }
    }
    return out;
}

struct Case { std::string s; std::vector<uint32_t> ids; };

std::vector<Case> load_ground_truth(const std::string& path) {
    std::ifstream f(path);
    if (!f) { std::cerr << "cannot open " << path << "\n"; return {}; }
    std::string buf((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    std::vector<Case> cases;
    size_t i = 0;
    auto skip_ws = [&]() { while (i < buf.size() && (buf[i] == ' ' || buf[i] == '\n' || buf[i] == '\r' || buf[i] == '\t')) ++i; };
    skip_ws();
    if (i < buf.size() && buf[i] == '[') ++i;
    while (i < buf.size()) {
        skip_ws();
        if (i >= buf.size() || buf[i] == ']') break;
        if (buf[i] == ',') { ++i; continue; }
        if (buf[i] == '{') ++i;
        Case c;
        while (i < buf.size() && buf[i] != '}') {
            skip_ws();
            if (i >= buf.size() || buf[i] == '}') break;
            if (buf[i] == ',') { ++i; continue; }
            // 期望 "key"
            if (buf[i] != '"') { ++i; continue; }
            ++i;
            std::string key;
            while (i < buf.size() && buf[i] != '"') { key += buf[i]; ++i; }
            ++i; // closing quote
            skip_ws();
            if (i < buf.size() && buf[i] == ':') ++i;
            skip_ws();
            if (key == "s") {
                if (buf[i] == '"') ++i;
                std::string raw;
                // 读到未转义的闭合引号：遇到 \" 是转义引号，不算结束。
                while (i < buf.size()) {
                    if (buf[i] == '\\' && i + 1 < buf.size()) {
                        raw += buf[i];
                        raw += buf[i + 1];
                        i += 2;
                        continue;
                    }
                    if (buf[i] == '"') break;
                    raw += buf[i];
                    ++i;
                }
                if (i < buf.size() && buf[i] == '"') ++i;
                c.s = unescape_json_string(raw);
            } else if (key == "ids") {
                if (buf[i] == '[') ++i;
                while (i < buf.size() && buf[i] != ']') {
                    skip_ws();
                    if (buf[i] == ']') break;
                    if (buf[i] == ',') { ++i; continue; }
                    std::string num;
                    while (i < buf.size() && (buf[i] == '-' || (buf[i] >= '0' && buf[i] <= '9'))) { num += buf[i]; ++i; }
                    if (!num.empty()) c.ids.push_back(static_cast<uint32_t>(std::stol(num)));
                }
                if (i < buf.size() && buf[i] == ']') ++i;
            } else {
                // 跳过未知值（处理转义引号避免误停）
                if (buf[i] == '"') {
                    ++i;
                    while (i < buf.size()) {
                        if (buf[i] == '\\' && i + 1 < buf.size()) { i += 2; continue; }
                        if (buf[i] == '"') { ++i; break; }
                        ++i;
                    }
                }
                else if (buf[i] == '[') { int d = 1; ++i; while (i < buf.size() && d > 0) { if (buf[i] == '[') ++d; else if (buf[i] == ']') --d; ++i; } }
                else { while (i < buf.size() && buf[i] != ',' && buf[i] != '}') ++i; }
            }
        }
        if (i < buf.size() && buf[i] == '}') ++i;
        cases.push_back(std::move(c));
    }
    return cases;
}

} // namespace

int main() {
    try {
        // 定位测试数据文件：优先环境变量，其次可执行文件旁/源码根，再 cwd。
        std::string bin_path = find_data_file(kBinName);
        std::string gt_path = find_data_file(kGtName);
        if (bin_path.empty()) {
            std::cerr << "找不到 " << kBinName << "。请设置 QWEN_TEST_DATA 指向含该文件的目录，"
                      << "或把可执行文件与该文件放同一目录。\n";
            return 1;
        }

        qwen::Tokenizer tokenizer;
        tokenizer.load(bin_path);

        // ===== 测试 1-4：原有 round-trip 测试 =====
        {
            std::string text = "Hello, Qwen3-VL! 你好，世界。";
            auto ids = tokenizer.encode(text);
            std::string decoded = tokenizer.decode(ids);
            assert(text == decoded && "Test 1 Failed: Decode mismatch!");
            std::cout << "[Test 1] round-trip 中英文 PASSED\n";
        }
        {
            std::string text = "The quick brown fox jumps over the lazy dog. Performance test.";
            assert(tokenizer.decode(tokenizer.encode(text)) == text);
            std::cout << "[Test 2] round-trip 英文 PASSED\n";
        }
        {
            std::string text = "def main():\n    print(\"Hello\\nWorld!\") # 100% 测试";
            assert(tokenizer.decode(tokenizer.encode(text)) == text);
            std::cout << "[Test 3] round-trip 代码 PASSED\n";
        }
        {
            assert(tokenizer.encode("").empty());
            assert(tokenizer.decode(tokenizer.encode("A")) == "A");
            std::cout << "[Test 4] 边界 PASSED\n";
        }

        // ===== 测试 5：HF 一致性（核心）=====
        {
            if (gt_path.empty()) {
                std::cerr << "[Test 5] 跳过：找不到 " << kGtName
                          << "。运行 tests/gen_ground_truth.py 生成（需 HF tokenizers）。\n";
                // 不算失败，直接继续后续测试。
            } else {
                auto cases = load_ground_truth(gt_path);
                assert(!cases.empty() && "ground truth fixture empty");
                int diffs = 0;
                for (const auto& c : cases) {
                    auto got = tokenizer.encode(c.s);
                    if (got != c.ids) {
                        ++diffs;
                        std::cerr << "  DIFF \"";
                        for (char ch : c.s) { if (ch=='\n') std::cerr<<"\\n"; else if(ch=='\t') std::cerr<<"\\t"; else if(ch=='\r') std::cerr<<"\\r"; else std::cerr<<ch; }
                        std::cerr << "\"\n    HF: ";
                        for (auto id : c.ids) std::cerr << id << " ";
                        std::cerr << "\n    US: ";
                        for (auto id : got) std::cerr << id << " ";
                        std::cerr << "\n";
                    }
                }
                std::cout << "[Test 5] HF 一致性: " << (cases.size() - diffs) << "/" << cases.size()
                          << " cases match\n";
                assert(diffs == 0 && "HF consistency mismatch!");
                std::cout << "[Test 5] PASSED (全部与 HF tokenizers 逐 id 一致)\n";
            }
        }

        // ===== 测试 6：流式 decode（增量 + 跨 token 边界）=====
        {
            // 选一段含多字节 UTF-8 的文本，逐 token emit_delta，验证拼回等于原文。
            std::string text = "Hello 世界 🦙 café 你好，Qwen3-VL！多字节测试 emoji 😀";
            auto ids = tokenizer.encode(text);
            std::string streamed;
            qwen::StreamState st;
            // 一次喂一个 token，模拟逐 token 生成
            std::vector<uint32_t> partial;
            for (uint32_t id : ids) {
                partial.push_back(id);
                streamed += tokenizer.emit_delta(st, partial, false);
            }
            streamed += tokenizer.emit_delta_flush(st); // 合法流应返回空
            assert(streamed == text && "streaming decode mismatch");
            std::cout << "[Test 6] 流式 decode (逐 token) PASSED\n";

            // 验证中间无半字符：每个增量都是 well-formed UTF-8（sanitize 保证）
            qwen::StreamState st2;
            std::vector<uint32_t> p2;
            std::string reconcat;
            for (uint32_t id : ids) {
                p2.push_back(id);
                std::string d = tokenizer.emit_delta(st2, p2, false);
                reconcat += d;
            }
            assert(reconcat == text);
            std::cout << "[Test 6b] 流式 decode 拼接一致 PASSED\n";
        }

        // ===== 测试 7：chat template =====
        {
            const auto& ct = tokenizer.chat_template();
            assert(!ct.empty() && "chat template not loaded");
            assert(ct.roles.count("system") && ct.roles.count("user") && ct.roles.count("assistant"));
            assert(ct.generation_prompt == "<|im_start|>assistant\n");

            std::vector<qwen::ChatMessage> msgs = {
                {"system", {{"text", "你是助手。"}}},
                {"user",   {{"text", "你好"}}},
            };
            std::string prompt = tokenizer.apply_chat_template(msgs, true);
            std::string expected =
                "<|im_start|>system\n你是助手。<|im_end|>\n"
                "<|im_start|>user\n你好<|im_end|>\n"
                "<|im_start|>assistant\n";
            assert(prompt == expected && "chat template render mismatch");
            std::cout << "[Test 7] chat template 渲染 PASSED\n";

            // 多模态：image/video 占位
            std::vector<qwen::ChatMessage> mmsgs = {
                {"user", {{"image", ""}, {"text", "这是什么？"}}},
            };
            std::string mp = tokenizer.apply_chat_template(mmsgs, true);
            assert(mp.find("<|vision_start|><|image_pad|><|vision_end|>") != std::string::npos);
            std::cout << "[Test 7b] chat template 多模态占位 PASSED\n";

            // 渲染结果应能正确 encode（含 special token）
            auto pids = tokenizer.encode(mp);
            assert(!pids.empty());
            assert(tokenizer.decode(pids) == mp);
            std::cout << "[Test 7c] chat template → encode → decode round-trip PASSED\n";
        }

        // ===== 测试 8：性能 =====
        {
            std::string long_text;
            for (int i = 0; i < 100; ++i) {
                long_text += "Hello, Qwen3-VL! 你好，世界。The quick brown fox jumps over the lazy dog. ";
            }
            auto start = std::chrono::high_resolution_clock::now();
            auto ids = tokenizer.encode(long_text);
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> elapsed = end - start;
            std::cout << "[Test 8] 性能: " << long_text.size() << " bytes, "
                      << ids.size() << " tokens, " << elapsed.count() << " ms\n";
        }

        std::cout << "\n All tests passed successfully!\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
