// Copyright 2025 @junka
#include <string>
#include <cstdint>
#include <cstdio>
#include <cstring>

#include "load.hpp"

namespace vkop {
namespace load {

VkModel::VkModel(const std::string& filePath) {
    loadFromBinary(filePath);
}

void VkModel::loadFromBinary(const std::string& filePath) {
    file_mapping_ = std::make_unique<FileMapping>();
    if (!file_mapping_->map_file(filePath)) {
        file_mapping_.reset();
        throw std::runtime_error("Failed to map file: " + filePath);
    }

    const auto* ptr = static_cast<const uint8_t*>(file_mapping_->data);
    size_t size = file_mapping_->size;
    if (size == 0 || ptr == nullptr) {
        throw std::runtime_error("Empty model file: " + filePath);
    }

    // The FlatBuffers file identifier ("VKOP") is required. Old struct.pack
    // files (no identifier) are rejected with a clear error so callers
    // reconvert with the FlatBuffers writer.
    if (!vkop::model::ModelBufferHasIdentifier(ptr)) {
        file_mapping_.reset();
        throw std::runtime_error(
            "Unrecognized model file (no VKOP identifier). Reconvert with "
            "`python3 -m onnx2vkop.cli -i <model>.onnx`: " + filePath);
    }
    loadFromFlatbuffer(ptr, size);
}

// ---------------------------------------------------------------------------
// FlatBuffers reader (zero-copy mmap view)
// ---------------------------------------------------------------------------
void VkModel::loadFromFlatbuffer(const uint8_t* buf, size_t size) {
    // Large models (the LLM vkopbin is 3.4GB) exceed FlatBuffers' default
    // max_size (INT32_MAX ~ 2.1GB) and would trip the verifier's constructor
    // assertion. Raise the cap to the actual buffer size so verification of
    // multi-GB mmaps works.
    ::flatbuffers::Verifier::Options vopts;
    vopts.max_size = size + 1;
    ::flatbuffers::Verifier verifier(buf, size, vopts);
    if (!vkop::model::VerifyModelBuffer(verifier)) {
        throw std::runtime_error("Invalid VKOP model: FlatBuffers verification failed");
    }

    const auto* model = vkop::model::GetModel(buf);
    if (model->version() != 3) {
        throw std::runtime_error(
            "Unsupported VKOP model version: " + std::to_string(model->version()) +
            " (expected 3: -1 dynamic sentinel + value_dynamic shape-cache "
            "annotation). Reconvert with "
            "`python3 -m onnx2vkop.cli -i <model>.onnx`.");
    }

    // Kernel plans: the converter's kernel selection, read before the nodes so
    // Node.plan can be range-checked against them. A file written by an older
    // writer has none, and a node's plan then stays -1 — the built-in OpType
    // table, which is what the runtime used for every model until now.
    if (model->kernel_plans()) {
        auto fbstr = [](const ::flatbuffers::String *s) {
            return s ? s->str() : std::string();
        };
        this->kernel_plans.reserve(model->kernel_plans()->size());
        for (uint32_t i = 0; i < model->kernel_plans()->size(); ++i) {
            const auto *p = model->kernel_plans()->Get(i);
            vkop::core::KernelPlan plan;
            plan.op_type = fbstr(p->op_type());
            plan.image = fbstr(p->image());
            plan.image_fp16 = fbstr(p->image_fp16());
            plan.buffer = fbstr(p->buffer());
            plan.buffer_fp16 = fbstr(p->buffer_fp16());
            if (p->variants()) {
                plan.variants.reserve(p->variants()->size());
                for (uint32_t v = 0; v < p->variants()->size(); ++v) {
                    const auto *s = p->variants()->Get(v);
                    if (s) {
                        plan.variants.push_back(s->str());
                    }
                }
            }
            this->kernel_plans.push_back(std::move(plan));
        }
    }

    // inputs / outputs
    auto read_shapes = [](const ::flatbuffers::Vector<::flatbuffers::Offset<vkop::model::ShapeRef>>* v) {
        std::vector<Shape> out;
        if (v) {
            out.reserve(v->size());
            for (uint32_t i = 0; i < v->size(); ++i) {
                const auto* s = v->Get(i);
                Shape shape;
                shape.name = s->name() ? s->name()->str() : "";
                shape.dtype = s->dtype() ? s->dtype()->str() : "";
                shape.value_dynamic = s->value_dynamic();
                if (s->dims()) {
                    shape.dims.reserve(s->dims()->size());
                    for (uint32_t d = 0; d < s->dims()->size(); ++d) {
                        shape.dims.push_back(s->dims()->Get(d));
                    }
                }
                out.push_back(std::move(shape));
            }
        }
        return out;
    };

    this->inputs = read_shapes(model->inputs());
    this->outputs = read_shapes(model->outputs());

    // nodes
    if (model->nodes()) {
        this->nodes.reserve(model->nodes()->size());
        for (uint32_t i = 0; i < model->nodes()->size(); ++i) {
            const auto* n = model->nodes()->Get(i);
            Node node;
            node.op_type = n->op_type() ? n->op_type()->str() : "";
            node.name = n->name() ? n->name()->str() : "";

            if (n->attributes()) {
                for (uint32_t a = 0; a < n->attributes()->size(); ++a) {
                    const auto* attr = n->attributes()->Get(a);
                    std::string key = attr->key() ? attr->key()->str() : "";
                    node.attributes.emplace(std::move(key), attrValueToString(attr));
                }
            }

            node.inputs = read_shapes(n->inputs());
            node.outputs = read_shapes(n->outputs());

            if (n->dependencies()) {
                for (uint32_t d = 0; d < n->dependencies()->size(); ++d) {
                    const auto* s = n->dependencies()->Get(d);
                    if (s) node.dependencies.insert(s->str());
                }
            }
            if (n->dependents()) {
                for (uint32_t d = 0; d < n->dependents()->size(); ++d) {
                    const auto* s = n->dependents()->Get(d);
                    if (s) node.dependents.insert(s->str());
                }
            }
            // Out-of-range is a malformed file, not a fatal one: the built-in
            // key table is still there, so warn and treat it as planless
            // rather than refusing a model that would otherwise run.
            const int plan_index = n->plan();
            if (plan_index >= 0 &&
                plan_index >= static_cast<int>(this->kernel_plans.size())) {
                std::fprintf(stderr,
                             "[load] node %s has plan=%d but the model carries "
                             "%zu plan row(s); falling back to the built-in "
                             "kernel table\n",
                             node.name.c_str(), plan_index,
                             this->kernel_plans.size());
            } else {
                node.plan_index = plan_index;
            }
            this->nodes.push_back(std::move(node));
        }
    }

    // initializer blob — zero-copy view straight into the mmap'd FlatBuffer.
    // For very large models the Python writer appends the blob as "external
    // data" after the FlatBuffer and stores its start offset in the last 8
    // bytes of the file (LE uint64); the FlatBuffer's own blob field is then
    // empty. Detect that trailer and point initializer_memory at the external
    // region instead.
    const uint8_t* blob_mem = nullptr;
    size_t blob_size = 0;
    const auto* blob = model->initializer_blob();
    if (blob) {
        blob_mem = blob->Data();
        blob_size = blob->size();
    }
    if ((blob_mem == nullptr || blob_size == 0) && size >= 8) {
        uint64_t external_offset = 0;
        std::memcpy(&external_offset, buf + size - 8, sizeof(uint64_t));
        if (external_offset > 0 && external_offset + 8 <= size) {
            blob_mem = buf + external_offset;
            blob_size = size - 8 - external_offset;
        }
    }
    if (blob_mem) {
        this->initializer_memory = blob_mem;
        this->initializer_memory_size = blob_size;
        // buf is the mmap base, which is file offset 0, so this pointer delta
        // is also the absolute offset to pread from.
        this->initializer_file_offset = static_cast<size_t>(blob_mem - buf);
    } else {
        this->initializer_memory = nullptr;
        this->initializer_memory_size = 0;
    }

    // Embedded SPIR-V archive (VKSP), written by onnx2vkop by default so a
    // single .vkopbin carries the shaders its graph dispatches. Same
    // zero-copy-view contract as the initializer blob: the pointer is into
    // the mmap and only valid while this VkModel lives. Runtime copies it
    // out (ShaderArchive::open_from_memory owns its bytes) because that
    // mapping is released when LoadModel returns.
    if (const auto* arch = model->shader_archive()) {
        if (arch->size() > 0) {
            this->shader_archive_memory = arch->Data();
            this->shader_archive_size = arch->size();
        }
    }

    // initializer entries — offsets come pre-computed from the Python writer.
    if (model->initializers()) {
        for (uint32_t i = 0; i < model->initializers()->size(); ++i) {
            const auto* e = model->initializers()->Get(i);
            Initializer init;
            init.name = e->name() ? e->name()->str() : "";
            init.dtype = e->dtype() ? e->dtype()->str() : "";
            init.size = static_cast<size_t>(e->size());
            // Absent in a pre-paging file; the accessors then return "" / -1,
            // which is precisely the no-paging case.
            init.moe_group = e->moe_group() ? e->moe_group()->str() : "";
            init.moe_expert_id = e->moe_expert_id();
            if (e->dims()) {
                init.dims.reserve(e->dims()->size());
                for (uint32_t d = 0; d < e->dims()->size(); ++d) {
                    init.dims.push_back(e->dims()->Get(d));
                }
            }
            this->initializer_offsets[init.name] = static_cast<size_t>(e->offset());
            this->initializers.emplace(init.name, std::move(init));
        }
    }

    printf("Initializer blob: %zu bytes, %zu entries\n",
           this->initializer_memory_size, this->initializers.size());

    // unified-tensor sub-allocation metadata (replaces the
    // unified_metadata/unified_names/unified_tensors magic-initializer hack).
    this->unified = model->unified();
    this->unified_blob_offset = static_cast<size_t>(model->unified_blob_offset());
    if (model->unified_names()) {
        this->unified_names = model->unified_names()->str();
    }
    if (model->unified_meta()) {
        this->unified_meta.reserve(model->unified_meta()->size());
        for (uint32_t i = 0; i < model->unified_meta()->size(); ++i) {
            this->unified_meta.push_back(*model->unified_meta()->Get(i));
        }
    }

    // RGBA conversion metadata — on load, rewrite the affected initializers'
    // dims to the 4-D RGBA shape (mirrors the legacy load.cpp behaviour) so
    // runtime's copyToGPUImage sees the right dimensions.
    this->rgba = model->rgba();
    if (model->rgba_names()) {
        this->rgba_names = model->rgba_names()->str();
    }
    if (model->rgba_meta()) {
        this->rgba_meta.reserve(model->rgba_meta()->size());
        for (uint32_t i = 0; i < model->rgba_meta()->size(); ++i) {
            this->rgba_meta.push_back(*model->rgba_meta()->Get(i));
        }

        size_t name_idx_offset = 0;
        for (const auto& meta : this->rgba_meta) {
            const char* base = this->rgba_names.data();
            std::string name(base + name_idx_offset, base + name_idx_offset + meta.name_len());
            auto it = this->initializers.find(name);
            if (it != this->initializers.end()) {
                it->second.dims.resize(4);
                for (int i = 0; i < 4; ++i) {
                    it->second.dims[i] = static_cast<uint32_t>(meta.dims()->Get(i));
                }
            }
            name_idx_offset += meta.name_len();
        }
    }

    // concurrent execution levels
    if (model->concurrent_levels()) {
        this->concurrent_execution_levels.reserve(model->concurrent_levels()->size());
        for (uint32_t i = 0; i < model->concurrent_levels()->size(); ++i) {
            const auto* lvl = model->concurrent_levels()->Get(i);
            std::vector<std::string> level;
            if (lvl && lvl->nodes()) {
                level.reserve(lvl->nodes()->size());
                for (uint32_t j = 0; j < lvl->nodes()->size(); ++j) {
                    const auto* s = lvl->nodes()->Get(j);
                    if (s) level.push_back(s->str());
                }
            }
            this->concurrent_execution_levels.push_back(std::move(level));
        }
    }
}

std::string VkModel::attrValueToString(const vkop::model::Attribute* attr) {
    switch (attr->type()) {
        case vkop::model::AttrType_String:
            return attr->sval() ? attr->sval()->str() : "";
        case vkop::model::AttrType_Int64:
            return std::to_string(attr->ival());
        case vkop::model::AttrType_Float32:
            return std::to_string(attr->fval());
        case vkop::model::AttrType_Bool:
            return attr->bval() ? "1" : "0";
        case vkop::model::AttrType_Ints: {
            // Schema stores [uint], but ops semantically need signed ints
            // (e.g. ReduceMean axes=[-1]). The converter writes negatives as
            // two's-complement uint32; reinterpret back. All previously
            // emitted binaries only stored 0..2^31-1 here, so this is a
            // no-op for them.
            const auto* v = attr->ints();
            if (!v || v->size() == 0) return "[]";
            std::string value =
                "[" + std::to_string(static_cast<int32_t>(v->Get(0)));
            for (uint32_t j = 1; j < v->size(); ++j) {
                value += ", " +
                         std::to_string(static_cast<int32_t>(v->Get(j)));
            }
            return value + "]";
        }
        case vkop::model::AttrType_Floats: {
            const auto* v = attr->floats();
            if (!v || v->size() == 0) return "[]";
            std::string value = "[" + std::to_string(v->Get(0));
            for (uint32_t j = 1; j < v->size(); ++j) {
                value += ", " + std::to_string(v->Get(j));
            }
            return value + "]";
        }
        case vkop::model::AttrType_Tensor: {
            const auto* t = attr->tval();
            if (!t) return "";
            std::string dtype = t->dtype() ? t->dtype()->str() : "";
            std::string value = dtype + "[";
            if (t->dims()) {
                for (uint32_t j = 0; j < t->dims()->size(); ++j) {
                    if (j) value += ", ";
                    value += std::to_string(t->dims()->Get(j));
                }
            }
            value += "]";
            // 常量折叠把大/向量常量固化成 Tensor 属性；很多节点（如
            // Constant -> Mul/Add 的 2048/6144 维 bias 向量）只需把这
            // 段字节搬到 CPU 端临时张量即可，不必落到 initializer blob。
            // 这里不保留 tensor 数据本身——当前 runtime 只消费标量字符串
            // 属性；真正的 TensorData 由后续的 Constant-op 运行时处理。
            return value;
        }
        default:
            throw std::runtime_error("Unknown attribute type tag");
    }
}



} // namespace load
} // namespace vkop
