#include "tokenizer.hpp"
#include <iostream>
#include <cassert>
#include <chrono>

int main() {
    try {
        qwen::Tokenizer tokenizer;
        tokenizer.load("../qwen3_vl.bin");

        // 测试用例 1: 基础中英文混合
        {
            std::string text = "Hello, Qwen3-VL! 你好，世界。";
            auto ids = tokenizer.encode(text);
            std::string decoded = tokenizer.decode(ids);
            std::cout << "[Test 1] Original: " << text << "\n";
            std::cout << "[Test 1] Decoded:  " << decoded << "\n";
            assert(text == decoded && "Test 1 Failed: Decode mismatch!");
            std::cout << "[Test 1] PASSED\n\n";
        }

        // 测试用例 2: 纯英文长文本
        {
            std::string text = "The quick brown fox jumps over the lazy dog. This is a performance test for BPE merge algorithm.";
            auto ids = tokenizer.encode(text);
            std::string decoded = tokenizer.decode(ids);
            assert(text == decoded && "Test 2 Failed: Decode mismatch!");
            std::cout << "[Test 2] Pure English PASSED\n\n";
        }

        // 测试用例 3: 特殊字符与代码片段
        {
            std::string text = "def main():\n    print(\"Hello\\nWorld!\") # 100% 测试";
            auto ids = tokenizer.encode(text);
            std::string decoded = tokenizer.decode(ids);
            assert(text == decoded && "Test 3 Failed: Decode mismatch!");
            std::cout << "[Test 3] Code & Special Chars PASSED\n\n";
        }

        // 测试用例 4: 极端边界 (空字符串 & 单字符)
        {
            auto ids1 = tokenizer.encode("");
            assert(ids1.empty() && "Test 4 Failed: Empty string should yield empty tokens!");
            
            auto ids2 = tokenizer.encode("A");
            std::string decoded2 = tokenizer.decode(ids2);
            assert(decoded2 == "A" && "Test 4 Failed: Single char mismatch!");
            std::cout << "[Test 4] Edge Cases PASSED\n\n";
        }

        // 测试用例 5: 性能压测 (Benchmark)
        {
            std::string long_text;
            for (int i = 0; i < 100; ++i) {
                long_text += "Hello, Qwen3-VL! 你好，世界。The quick brown fox jumps over the lazy dog. ";
            }
            
            auto start = std::chrono::high_resolution_clock::now();
            auto ids = tokenizer.encode(long_text);
            auto end = std::chrono::high_resolution_clock::now();
            
            std::chrono::duration<double, std::milli> elapsed = end - start;
            std::cout << "[Test 5] Performance Benchmark:\n";
            std::cout << "  Text Length: " << long_text.size() << " bytes\n";
            std::cout << "  Tokens Generated: " << ids.size() << "\n";
            std::cout << "  Encode Time: " << elapsed.count() << " ms\n";
            std::cout << "[Test 5] PASSED\n\n";
        }
        std::cout << " All tests passed successfully!\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
    }
    return 0;
}