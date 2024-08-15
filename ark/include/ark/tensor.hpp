// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_TENSOR_HPP
#define ARK_TENSOR_HPP

#include <ark/data_type.hpp>
#include <ark/dims.hpp>
#include <ark/model_ref.hpp>
#include <ostream>

namespace ark {

///
/// Tensor is a view of a memory space.
///
/// Illustration of a single axis of a tensor:
///
/// 0         offset                                                      stride
/// |------------|-------------shape-------------|---------------------------|
///               <----------------------------->
///                  data range of this tensor
///
class Tensor {
   protected:
    friend class Model;
    ModelTensorRef ref_;

   public:
    Tensor() = default;
    Tensor(ModelTensorRef ref) : ref_(ref) {}
    Tensor(const Tensor &other) = default;
    Tensor &operator=(const Tensor &other) = default;
    Tensor(void *data_ptr, int32_t device_id, const std::vector<int64_t> &shape,
           const DataType &dtype);

    bool operator==(const Tensor &other) const { return ref_ == other.ref_; }
    bool operator!=(const Tensor &other) const { return ref_ != other.ref_; }

    bool is_null() const { return !ref_; }

    ModelTensorRef ref() const { return ref_; }

    size_t id() const;

    Dims shape() const;

    Dims strides() const;

    Dims offsets() const;

    Dims padded_shape() const;

    const DataType &data_type() const;

    Dims torch_strides() const;

    friend struct std::hash<Tensor>;
};

const Tensor NullTensor;

std::ostream &operator<<(std::ostream &os, const Tensor &tensor);

}  // namespace ark

namespace std {
template <>
struct hash<const ark::Tensor> {
    size_t operator()(const ark::Tensor &t) const {
        return hash<size_t>()(t.id());
    }
};
}  // namespace std

#endif  // ARK_TENSOR_HPP
