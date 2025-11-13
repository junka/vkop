#include "vulkan/VulkanDevice.hpp"
#include "vulkan/VulkanInstance.hpp"
#include "include/logger.hpp"
#include "core/Tensor.hpp"
#include "model/load.hpp"
#include "ops/OperatorFactory.hpp"
#include "ops/Ops.hpp"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <random>
#include <chrono>
#include <cmath>
#include <unordered_set>

#include <sys/types.h>
#include <unistd.h>
#include <vulkan/vulkan_core.h>

using vkop::VulkanInstance;
using vkop::VulkanDevice;
using vkop::core::Tensor;
using vkop::core::ITensor;
using vkop::load::VkModel;
using vkop::ops::OperatorFactory;


enum class Category : int {
    UNKNOWN = 0,
    HUMAN_FACE = 1,
    LICENSE_PLATE = 2,
    OTHERS = 999
};

using MaskInfo = struct MaskInfo {
    float x1;
    float y1;
    float x2;
    float y2;

    float score;
    Category category;
};

static float score_threshold[2] = {0.35F, 0.5F};

std::vector<MaskInfo> postProcessNMS(
    const float* hm_data, const float* hm_nms_data,
    const float* reg_data, const float* dim_data,
    const float* cls_data, int H, int W, int image_h, int image_w,
    int tensor_h, int tensor_w
) {
    auto sigmoid = [](float x) -> float {
        return 1.0f / (1.0f + std::exp(-x));
    };
    const int num_points = H * W;

    std::vector<int> indices(num_points);
    for (int i = 0; i < num_points; i++) {
        indices[i] = i;
    }
    std::partial_sort(indices.begin(), indices.begin() + 64, indices.end(),
        [hm_nms_data](const int& i1, const int& i2) {
            return hm_nms_data[i1] > hm_nms_data[i2];
        });

    std::vector<MaskInfo> detections;
    for (int i = 0; i < 64; i++) {
        int idx = indices.at(i);

        if (std::fabs(hm_data[idx] - hm_nms_data[idx]) > 1e-6F) {
            continue;
        }

        float score = sigmoid(hm_nms_data[idx]);
        int category = 0;
        const float cls_value1 = cls_data[idx];
        const float cls_value2 = cls_data[num_points + idx];
        if (cls_value1 > cls_value2) {
            category = 0;
        } else {
            category = 1;
        }

        if (score < score_threshold[category]) {
            continue;
        }

        int y = idx / W;
        int x = idx % W;
        float xo = sigmoid(reg_data[idx]) + x;
        float yo = sigmoid(reg_data[num_points + idx]) + y;
        // std::cout << "point at index " << idx << " has (x, y): (" << xo << ", " << yo << ")" << std::endl;
        xo *= (tensor_w / W);
        yo *= (tensor_h / H);

        float wo = std::exp(dim_data[idx]) * tensor_w / W;
        float ho = std::exp(dim_data[num_points + idx]) * tensor_h / H;

        // std::cout << "index " << idx << " score " << score <<" Raw bbox (center x, center y, width, height): (" << xo << ", " << yo << ", " << wo << ", " << ho << ")" << std::endl;
        xo = xo * W / tensor_w;
        wo = wo * W / tensor_w;
        yo = yo * H / tensor_h;
        ho = ho * H / tensor_h;

        float scale_x = static_cast<float>(tensor_w) / W;
        float scale_y = static_cast<float>(tensor_h) / H;

        float x1 = (xo - wo * 0.5F) * scale_x * image_w / tensor_w;
        float y1 = (yo - ho * 0.5F) * scale_y * image_h / tensor_h;
        float x2 = (xo + wo * 0.5F) * scale_x * image_w / tensor_w;
        float y2 = (yo + ho * 0.5F) * scale_y * image_h / tensor_h;
        x1 = std::max(0, static_cast<int>(x1));
        y1 = std::max(0, static_cast<int>(y1));
        x2 = std::min(static_cast<int>(x2), image_w - 1);
        y2 = std::min(static_cast<int>(y2), image_h - 1);

        std::cout << "Detected plate at: (" << x1 << ", " << y1 << ") to (" << x2 << ", " << y2 << ") with score: " << score
                  << " and category: " << category << std::endl;

        // detections.push_back({x1, y1, x2, y2, score, Category::category});
    }
    return detections;
}



void resize_YUV(std::vector<uint8_t> raw_image, int image_h, int image_w, std::shared_ptr<Tensor<float>> &t) {
    int in_h = t->getTensorShape()[2];
    int in_w = t->getTensorShape()[3];

    float* data_ptr = t->data();

    float x_ratio = float(image_w - 1) / (in_w - 1);
    float y_ratio = float(image_h - 1) / (in_h - 1);

    const uint8_t* y_src = raw_image.data();
    const uint8_t* u_src = raw_image.data() + image_w * image_h;
    const uint8_t* v_src = raw_image.data() + 2 * image_w * image_h;

    float* y_dst = data_ptr;
    float* u_dst = data_ptr + in_w * in_h;
    float* v_dst = data_ptr + 2 * in_w * in_h;

    for (int dy = 0; dy < in_h; ++dy) {
        for (int dx = 0; dx < in_w; ++dx) {
            float src_x = dx * x_ratio;
            float src_y = dy * y_ratio;
            int x1 = static_cast<int>(src_x);
            int y1 = static_cast<int>(src_y);
            int x2 = std::min(x1 + 1, image_w - 1);
            int y2 = std::min(y1 + 1, image_h - 1);

            float dx_ratio = src_x - x1;
            float dy_ratio = src_y - y1;

            // 对 Y, U, V 分量分别进行双线性插值
            auto interpolate = [](const uint8_t* plane, int w, int x1, int y1, int x2, int y2, float dx, float dy) {
                uint8_t p11 = plane[y1 * w + x1];
                uint8_t p12 = plane[y1 * w + x2];
                uint8_t p21 = plane[y2 * w + x1];
                uint8_t p22 = plane[y2 * w + x2];
                return static_cast<uint8_t>(
                    p11 * (1 - dx) * (1 - dy) +
                    p12 * dx * (1 - dy) +
                    p21 * (1 - dx) * dy +
                    p22 * dx * dy
                );
            };

            int dst_idx = dy * in_w + dx;
            y_dst[dst_idx] = interpolate(y_src, image_w, x1, y1, x2, y2, dx_ratio, dy_ratio) / 255.0F;
            u_dst[dst_idx] = interpolate(u_src, image_w, x1, y1, x2, y2, dx_ratio, dy_ratio) / 255.0F;
            v_dst[dst_idx] = interpolate(v_src, image_w, x1, y1, x2, y2, dx_ratio, dy_ratio) / 255.0F;
        }
    }
}

int main(int argc, char *argv[]) {
    Logger::getInstance().setLevel(LOG_INFO);
    Logger::getInstance().enableFileOutput("log", true);
    VkPhysicalDevice phydev = VK_NULL_HANDLE;
    std::shared_ptr<VulkanDevice> dev;
    try {
        auto phydevs = VulkanInstance::getVulkanInstance().getPhysicalDevices();
        for (auto *pdev : phydevs) {
            dev = std::make_shared<VulkanDevice>(pdev);
            if (dev->getDeviceName().find("llvmpipe") != std::string::npos) {
                continue;
            }
            phydev = pdev;
            LOG_INFO("%s",dev->getDeviceName().c_str());
        }
    } catch (const std::exception &e) {
        LOG_ERROR("%s", e.what());
        return EXIT_FAILURE;
    }
    auto *device = dev->getLogicalDevice();
    auto cmdpool = std::make_shared<vkop::VulkanCommandPool>(device, dev->getComputeQueueFamilyIndex());

    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <binary_file_path> <image.yuv>" << std::endl;
        return 1;
    }
    try {
        std::string binary_file_path = argv[1];
        VkModel model(binary_file_path);
        // VkModel::dump_model(model);
        std::string image_file_path = argv[2];
        std::vector<uint8_t> frame;

        std::ifstream infile(image_file_path, std::ios::in | std::ios::binary);
        infile.seekg(0, std::ios::end);
        size_t file_size = infile.tellg();
        infile.seekg(0, std::ios::beg);
        frame.resize(file_size);
        infile.read(reinterpret_cast<char*>(frame.data()), file_size);
        infile.close();

        int image_h = 1080;
        int image_w = 1920;

        std::vector<std::shared_ptr<ITensor>> inputs;
        std::vector<std::shared_ptr<ITensor>> outputs;
        std::unordered_set<std::shared_ptr<ITensor>> output_tensor_set;
        std::unordered_map<std::string, std::shared_ptr<ITensor>> tensor_map;
        std::unordered_map<std::shared_ptr<ITensor>, std::string> tensor_name_map;

        std::vector<std::unique_ptr<vkop::ops::Operator>> ops_all;
        std::vector<std::vector<std::shared_ptr<ITensor>>> inputs_all;
        std::vector<std::vector<std::shared_ptr<ITensor>>> outputs_all;
        for (const auto& i : model.inputs) {
            auto t = std::make_shared<Tensor<float>>(i.dims);
            inputs.push_back(t);
            tensor_map[i.name] = t;
            tensor_name_map[t] = i.name;
        }

        for (const auto& o: model.outputs) {
            auto t = std::make_shared<Tensor<float>>(o.dims);
            t->toGPU();
            outputs.push_back(t);
            output_tensor_set.insert(t);
            tensor_map[o.name] = t;
            tensor_name_map[t] = o.name;
            std::cout << "create output tensor " << o.name << std::endl;
        }

        for (const auto& n: model.nodes) {
            auto t = vkop::ops::convert_opstring_to_enum(n.op_type);
            if (t == vkop::ops::OpType::CONSTANT || t == vkop::ops::OpType::UNKNOWN) {
                // make it as input for next ops
                continue;
            }
            std::vector<std::shared_ptr<ITensor>> node_inputs;
            std::vector<std::shared_ptr<ITensor>> node_outputs;

            for (const auto& out_shape : n.outputs) {
                if (tensor_map.find(out_shape.name) != tensor_map.end()) {
                    printf("find output tensor %s for op %s\n", out_shape.name.c_str(), n.op_type.c_str());
                    tensor_map[out_shape.name]->toGPU();
                    node_outputs.push_back(tensor_map[out_shape.name]);
                } else {
                    printf("make tensor on GPU %s\n", out_shape.name.c_str());
                    auto t = std::make_shared<Tensor<float>>(out_shape.dims);
                    t->toGPU();
                    tensor_map[out_shape.name] = t;
                    tensor_name_map[t] = out_shape.name;
                    node_outputs.push_back(t);
                }
            }
            for (const auto& in_shape : n.inputs) {
                if (tensor_map.find(in_shape.name) != tensor_map.end()) {
                    node_inputs.push_back(tensor_map[in_shape.name]);
                    std::cout << "find input tensor " << in_shape.name << " for op " << n.op_type << " on " << (tensor_map[in_shape.name]->is_on_GPU()? "GPU" : "CPU")<< std::endl;
                } else {
                    std::cout << "create empty tensor " << in_shape.name << " for op " << n.op_type << std::endl;
                    if (in_shape.dims.empty()) {
                        node_inputs.push_back(nullptr);
                        continue;
                    }
                    if (model.initializers.find(in_shape.name) != model.initializers.end()) {
                        auto& init = model.initializers.at(in_shape.name);
                        if (init.dims != in_shape.dims) {
                            throw std::runtime_error("Initializer dims do not match for " + in_shape.name);
                        }
                        if (init.dtype == "int64") {
                            // printf("load int64 initializer %s for op %s\n", in_shape.name.c_str(), n.op_type.c_str());
                            auto t = std::make_shared<Tensor<int64_t>>(in_shape.dims);
                            for (int i = 0; i < t->num_elements(); ++i) {
                                t->data()[i] = static_cast<float>(init.dataii[i]);
                            }
                            tensor_map[in_shape.name] = t;
                            tensor_name_map[t] = in_shape.name;
                            node_inputs.push_back(t);
                            // std::cout << "load int64 initializer " << in_shape.name << " for op " << n.op_type << std::endl;
                        } else if (init.dtype == "int32") {
                            auto t = std::make_shared<Tensor<int>>(in_shape.dims);
                            for (int i = 0; i < t->num_elements(); ++i) {
                                t->data()[i] = static_cast<float>(init.dataii[i]);
                            }
                            tensor_map[in_shape.name] = t;
                            tensor_name_map[t] = in_shape.name;
                            node_inputs.push_back(t);
                            // std::cout << "load int32 initializer " << in_shape.name << " for op " << n.op_type << std::endl;
                        } else if (init.dtype == "float32") {
                            auto t = std::make_shared<Tensor<float>>(in_shape.dims);
                            std::memcpy(t->data(), init.dataf.data(), t->num_elements() * sizeof(float));
                            tensor_map[in_shape.name] = t;
                            tensor_name_map[t] = in_shape.name;
                            node_inputs.push_back(t);
                            // std::cout << "load float32 initializer " << in_shape.name << " for op " << n.op_type << std::endl;
                        } else {
                            throw std::runtime_error("Only float32 initializer is supported for now " + init.dtype);
                        }
                    }
                }
            }

            auto op = OperatorFactory::get_instance().create(t);
            if (!op) {
                std::cout << "Fail to create operator" << std::endl;
                return 1;
            }

            op->set_runtime_device(phydev, dev, cmdpool);
            if (!n.attributes.empty()) {
                op->setAttribute(n.attributes);
            }
            ops_all.push_back(std::move(op));
            inputs_all.push_back(node_inputs);
            outputs_all.push_back(node_outputs);
        }

        auto t = vkop::core::as_tensor<float>(inputs[0]);
        resize_YUV(frame, image_h, image_w, t);
        t->printTensorShape();
        auto start = std::chrono::steady_clock::now();
        for (size_t i = 0; i < ops_all.size(); ++i) {
            printf("ops %s input tensors %ld\n", vkop::ops::convert_openum_to_string(ops_all[i]->get_type()).c_str(), inputs_all[i].size());
            ops_all[i]->apply(inputs_all[i], outputs_all[i]);
            for (auto &p: inputs_all[i]) {
                if (!p || p->num_dims() < 3) {
                    continue;
                }
                if (p->is_on_GPU()) {
                    printf("tensor %s already on GPU\n", tensor_name_map[p].c_str());
                } else {
                    printf("tensor %s on CPU\n", tensor_name_map[p].c_str());
                    auto t = vkop::core::as_tensor<float>(p);
                    t->copyToGPU(dev, cmdpool);
                }
            }
            ops_all[i]->execute(inputs_all[i], outputs_all[i]);
        }
        for (auto &p : outputs) {
            auto t = vkop::core::as_tensor<float>(p);
            t->copyToCPU(dev, cmdpool);
        }
        auto hm = vkop::core::as_tensor<float>(tensor_map["hm"]);
        auto reg = vkop::core::as_tensor<float>(tensor_map["reg"]);
        auto dim = vkop::core::as_tensor<float>(tensor_map["dim"]);
        auto cls = vkop::core::as_tensor<float>(tensor_map["cls"]);
        auto hm_nms = vkop::core::as_tensor<float>(tensor_map["hm_nms"]);

        postProcessNMS(hm->data(), hm_nms->data(), reg->data(), dim->data(), cls->data(), hm_nms->getTensorShape()[2], hm_nms->getTensorShape()[3], image_h, image_w,
            t->getTensorShape()[2], t->getTensorShape()[3]);
        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        std::cout << "inference time:" << elapsed.count() << " s" << std::endl;
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
        return 1;
    }
    // sleep(20000);
    return EXIT_SUCCESS;
}