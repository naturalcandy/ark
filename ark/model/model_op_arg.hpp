// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_MODEL_OP_ARG_HPP_
#define ARK_MODEL_OP_ARG_HPP_

#include <any>
#include <ostream>
#include <sstream>

#include "ark/dims.hpp"
#include "ark/model_ref.hpp"
#include "model_json.hpp"
#include "model_named_type.hpp"
#include "model_offset.hpp"

namespace ark {

template <typename T>
class ModelOpArgTName;

#define REGISTER_MODEL_OP_ARG_TYPE(_name, _type)              \
    template <>                                               \
    class ModelOpArgTName<_type> {                            \
       public:                                                \
        ModelOpArgTName() : name(#_name), type_str(#_type){}; \
        const std::string name;                               \
        const std::string type_str;                           \
    };

class ModelOpArg : public ModelNamedT {
   public:
    ModelOpArg();

    template <typename T>
    ModelOpArg(T val)
        : ModelNamedT(ModelOpArgTName<T>().name),
          type_str_(ModelOpArgTName<T>().type_str),
          val_(val) {}

    template <typename T>
    T value() const {
        return std::any_cast<T>(val_);
    }

    const std::string &type_str() const { return type_str_; }

    Json serialize() const;

    static ModelOpArg deserialize(const Json &serialized);

   private:
    std::string type_str_;
    std::any val_;
};

REGISTER_MODEL_OP_ARG_TYPE(INT, int)
REGISTER_MODEL_OP_ARG_TYPE(UINT32, uint32_t)
REGISTER_MODEL_OP_ARG_TYPE(INT64, int64_t)
REGISTER_MODEL_OP_ARG_TYPE(UINT64, uint64_t)
REGISTER_MODEL_OP_ARG_TYPE(BOOL, bool)
REGISTER_MODEL_OP_ARG_TYPE(FLOAT, float)
REGISTER_MODEL_OP_ARG_TYPE(DIMS, Dims)
REGISTER_MODEL_OP_ARG_TYPE(TENSOR, ModelTensorRef)
REGISTER_MODEL_OP_ARG_TYPE(OFFSET, ModelOffset)

}  // namespace ark

#endif  // ARK_MODEL_OP_ARG_HPP_
