// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "include/ark.h"
#include "include/ark_utils.h"
#include "ops_test_common.h"
#include "unittest/unittest_utils.h"
#include <cmath>

template <typename T>
void baseline_sqrt(std::vector<void *> &outputs,
                   const std::vector<ark::Dims> &output_shapes,
                   const std::vector<void *> &inputs,
                   const std::vector<ark::Dims> &)
{
    T *out = static_cast<T *>(outputs[0]);
    T *input = static_cast<T *>(inputs[0]);
    ark::Dims osh = output_shapes[0];
    for (ark::DimType i = 0; i < osh.size(); ++i) {
        out[i] = std::sqrt(input[i]);
    }
};

ark::unittest::State test_sqrt_fp32()
{
    ark::Model m;
    ark::Tensor *t = m.tensor(ark::Dims(4, 2, 1024), ark::FP32);
    ark::Tensor *out = m.sqrt(t);

    auto result =
        ark::op_test("sqrt_fp32", m, {t}, {out}, baseline_sqrt<float>);
    ark::op_test_log(result);
    return ark::unittest::SUCCESS;
}

int main()
{
    ark::init();
    UNITTEST(test_sqrt_fp32);
    return ark::unittest::SUCCESS;
}