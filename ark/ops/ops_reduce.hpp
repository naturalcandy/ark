// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_OPS_REDUCE_HPP_
#define ARK_OPS_REDUCE_HPP_

#include "model/model_op.hpp"

namespace ark {

class ModelOpReduce : public ModelOp {
   public:
    ModelOpReduce() = default;
    ModelOpReduce(const std::string &type_name, ModelTensorRef input, int axis,
                  bool keepdims, ModelTensorRef output);

    std::string impl_name(const Json &config) const override;

    std::vector<ModelOpArg> impl_args(const Json &config) const override;

    Json default_config(const ArchRef arch = ARCH_ANY) const override;
};

class ModelOpReduceMax : public ModelOpReduce {
   public:
    ModelOpReduceMax() = default;
    ModelOpReduceMax(ModelTensorRef input, int axis, bool keepdims,
                     ModelTensorRef output)
        : ModelOpReduce("ReduceMax", input, axis, keepdims, output) {}
};

class ModelOpReduceMean : public ModelOpReduce {
   public:
    ModelOpReduceMean() = default;
    ModelOpReduceMean(ModelTensorRef input, int axis, bool keepdims,
                      ModelTensorRef output)
        : ModelOpReduce("ReduceMean", input, axis, keepdims, output) {}
};

class ModelOpReduceSum : public ModelOpReduce {
   public:
    ModelOpReduceSum() = default;
    ModelOpReduceSum(ModelTensorRef input, int axis, bool keepdims,
                     ModelTensorRef output)
        : ModelOpReduce("ReduceSum", input, axis, keepdims, output) {}
};

}  // namespace ark

#endif  // ARK_OPS_REDUCE_HPP_
