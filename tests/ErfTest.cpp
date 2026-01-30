#include <vector>
#include <random>
#define _USE_MATH_DEFINES
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

double adaptive_simpsons(
    double (*f)(double),
    double a,
    double b,
    double eps,
    double max_recursion,
    double s,
    double fa,
    double fb,
    double fc,
    int depth
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

double integrate(double (*f)(double), double a, double b, double eps, double max_recursion) {
    double c = (a + b) / 2;
    double fa = f(a);
    double fb = f(b);
    double fc = f(c);

    double s = (b - a) / 6 * (fa + 4*fc + fb);
    return adaptive_simpsons(f, a, b, eps, max_recursion, s, fa, fb, fc, 0);
}

double erf(double x) {
    const double eps = 1e-8;
    const double max_recursion = 20;

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
    std::shared_ptr<Tensor<float>> output;

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
        input->reserveOnCPU();
        output = std::make_shared<Tensor<float>>(t);
        output->reserveOnCPU();

        std::random_device rd{};
        std::mt19937 gen{rd()};
        gen.seed(1024);
        std::normal_distribution<> input_dist{-3.0F, 6.0F};
        for (int i = 0; i < input->num_elements(); i++) {
            (*input)[i] = input_dist(gen);
            (*output)[i] = erf((*input)[i]);
        }
    }
};
}

int main() {
    Logger::getInstance().setLevel(LOG_INFO);
    Logger::getInstance().enableFileOutput("log", false);

    ErfTest erftest;
    if (!erftest.run_test<float>({erftest.input}, {erftest.output})) {
        return -1;
    }

    return 0;
}