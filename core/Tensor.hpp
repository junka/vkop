#include <vector>

namespace vkop {

class Tensor {
public:
    //empty
    Tensor();

    //nchw
    Tensor(int n, int c, int h, int w);

    // nchw in vector
    Tensor(std::vector<int> dims);
    

    // remove all assignment operator
    Tensor(const Tensor& tensor)  = delete;
    Tensor(const Tensor&& tensor) = delete;
    Tensor& operator=(const Tensor&) = delete;
    Tensor& operator=(const Tensor&&) = delete;
private:
    int n;
    int c;
    int h;
    int w;
};

}