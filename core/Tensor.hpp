#include <vector>
#include <cstdint>

namespace vkop {

class Tensor {
public:
    //empty
    Tensor();

    //nchw
    Tensor(int n, int c, int h, int w);

    // nchw in vector
    Tensor(std::vector<int> dims);

    // nchw in vector
    Tensor(std::vector<uint32_t> dims);
    

    void printTensor();
    /**
    * @brief Deleted copy constructor to prevent copying of tensor objects.
    */
    Tensor(const Tensor& tensor)  = delete;
    /**
    * @brief Deleted move constructor to prevent moving of tensor objects.
    */
    Tensor(const Tensor&& tensor) = delete;
    /**
    * @brief Deleted copy assignment operator to prevent copying of tensor objects.
    */
    Tensor& operator=(const Tensor&) = delete;

    /**
    * @brief Deleted move assignment operator to prevent moving of tensor objects.
    */
    Tensor& operator=(const Tensor&&) = delete;
private:
    int n;
    int c;
    int h;
    int w;
};

}