// Copyright 2025 @junka
#include <string>
#include <cstdint>
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
    ::flatbuffers::Verifier verifier(buf, size);
    if (!vkop::model::VerifyModelBuffer(verifier)) {
        throw std::runtime_error("Invalid VKOP model: FlatBuffers verification failed");
    }

    const auto* model = vkop::model::GetModel(buf);
    if (model->version() != 1) {
        throw std::runtime_error(
            "Unsupported VKOP model version: " + std::to_string(model->version()));
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
            this->nodes.push_back(std::move(node));
        }
    }

    // initializer blob — zero-copy view straight into the mmap'd FlatBuffer.
    const auto* blob = model->initializer_blob();
    if (blob) {
        this->initializer_memory = blob->Data();
        this->initializer_memory_size = blob->size();
    } else {
        this->initializer_memory = nullptr;
        this->initializer_memory_size = 0;
    }

    // initializer entries — offsets come pre-computed from the Python writer.
    if (model->initializers()) {
        for (uint32_t i = 0; i < model->initializers()->size(); ++i) {
            const auto* e = model->initializers()->Get(i);
            Initializer init;
            init.name = e->name() ? e->name()->str() : "";
            init.dtype = e->dtype() ? e->dtype()->str() : "";
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
            const auto* v = attr->ints();
            if (!v || v->size() == 0) return "[]";
            std::string value = "[" + std::to_string(v->Get(0));
            for (uint32_t j = 1; j < v->size(); ++j) {
                value += ", " + std::to_string(v->Get(j));
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
            return value;
        }
        default:
            throw std::runtime_error("Unknown attribute type tag");
    }
}



} // namespace load
} // namespace vkop
