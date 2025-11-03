#include <cstdint>
#include <vector>
#include <random>
#include <cmath>
#include <stack>

#include "setup.hpp"
#include "core/Tensor.hpp"
#include "include/logger.hpp"

using vkop::core::Tensor;
using vkop::tests::TestCase;

namespace {

double integrand(double t) {
    return exp(-t * t);
}

// 自适应辛普森积分法
double adaptive_simpsons(
    double (*f)(double),   // 被积函数
    double a,              // 积分下限
    double b,              // 积分上限
    double eps,            // 容许误差
    double max_recursion,  // 最大递归深度
    double s,              // 初始积分值
    double fa,             // f(a)
    double fb,             // f(b)
    double fc,             // f((a+b)/2)
    int depth              // 当前递归深度
) {
    struct Interval {
        double a, b, fa, fb, fc, s;
        int depth;
    };

    std::stack<Interval> stack;
    stack.push({a, b, fa, fb, fc, s, depth});

    double total = 0.0;

    while (!stack.empty()) {
        Interval current = stack.top();
        stack.pop();

        double c = (current.a + current.b) / 2;
        double h = current.b - current.a;
        double d = (current.a + c) / 2;
        double e = (c + current.b) / 2;

        double fd = f(d);
        double fe = f(e);

        double s_left = (h / 12) * (current.fa + 4 * fd + current.fc);
        double s_right = (h / 12) * (current.fc + 4 * fe + current.fb);
        double s_total = s_left + s_right;

        if (current.depth >= max_recursion || fabs(s_total - current.s) <= 15 * eps) {
            total += s_total + (s_total - current.s) / 15;
        } else {
            stack.push({c, current.b, current.fc, current.fb, fe, s_right, current.depth + 1});
            stack.push({current.a, c, current.fa, current.fc, fd, s_left, current.depth + 1});
        }
    }

    return total;
}

// 初始调用接口
double integrate(double (*f)(double), double a, double b, double eps, double max_recursion) {
    double c = (a + b) / 2;
    double fa = f(a);
    double fb = f(b);
    double fc = f(c);

    double s = (b - a) / 6 * (fa + 4*fc + fb);
    return adaptive_simpsons(f, a, b, eps, max_recursion, s, fa, fb, fc, 0);
}

// erf(x) 基于数值积分实现
double erf(double x) {
    const double eps = 1e-8;       // 控制积分误差
    const double max_recursion = 20; // 控制递归深度

    if (x == 0.0) {
        return 0.0;
    }

    double integral = integrate(integrand, 0.0, fabs(x), eps, max_recursion);
    double result = (2.0 / sqrt(M_PI)) * integral;

    return (x < 0.0) ? -result : result;
}

class ErfTest : public TestCase {
public:
    std::shared_ptr<Tensor<float>> input;
    std::vector<float> expectedOutput;

    ErfTest():TestCase("Erf") {
        initTestdata();
    }
private:
    void initTestdata()
    {
        std::vector<int> t = {
            1, 5, 64, 64
        };
        input = std::make_shared<Tensor<float>>(t);

        auto *input_ptr = input->data();
        expectedOutput.resize(input->num_elements());

        std::random_device rd{};
        std::mt19937 gen{rd()};
        gen.seed(1024);
        std::normal_distribution<> input_dist{-3.0F, 6.0F};
        for (int i = 0; i < input->num_elements(); i++) {
            input_ptr[i] = input_dist(gen);
            expectedOutput[i] = erf(input_ptr[i]);
        }
    }
};
}

int main() {
    Logger::getInstance().setLevel(LOG_INFO);
    Logger::getInstance().enableFileOutput("log", false);

    ErfTest erftest;
    if (!erftest.run_test({erftest.input}, erftest.expectedOutput)) {
        return -1;
    }

    return 0;
}