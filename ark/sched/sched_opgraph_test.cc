// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "ark.h"
#include "logging.h"
#include "sched_opgraph.h"
#include "unittest/unittest_utils.h"

ark::unittest::State test_sched_opgraph()
{
    ark::Model model;

    // Basic Test.
    // Model graph:
    //
    //   TensorOp --> t0 --+--> AddOp --> t2
    //                     |
    //   TensorOp --> t1 --+
    //                     |
    //   TensorOp --> tx --+  (tx is the output reference, hidden from the code)
    //

    ark::Tensor *t0 = model.tensor({1}, ark::FP32);
    ark::Tensor *t1 = model.tensor({1}, ark::FP32);
    ark::Tensor *t2 = model.add(t0, t1);
    UNITTEST_TRUE(model.verify());

    // OpNode graph (parentheses indicate a OpNode):
    //
    //   (AddOp,)
    //

    ark::OpGraph graph(model);
    UNITTEST_EQ(graph.get_nodes().size(), 1UL);

    auto node = graph.get_nodes().front().get();
    UNITTEST_EQ(node->ops.size(), 1UL);
    UNITTEST_EQ(node->ops[0]->outputs[0], t2);
    UNITTEST_EQ(node->ops[0]->inputs[0], t0);
    UNITTEST_EQ(node->ops[0]->inputs[1], t1);
    UNITTEST_EQ(node->users.size(), 0UL);
    UNITTEST_EQ(node->producers.size(), 0UL);

    // Test a chain of Ops that share an input tensor.
    // Model graph:
    //
    // TensorOp --> t0 --+--> AddOp --> t2 ------+--> AddOp --> t3
    //                   |                       |
    // TensorOp --> t1 --+-----------------------+
    //                   |                       |
    // TensorOp --> tx --+     TensorOp --> ty --+
    //
    // (tx and ty are output references, hidden from the code)
    //

    ark::Tensor *t3 = model.add(t2, t1);
    UNITTEST_TRUE(model.verify());

    // OpNode graph (parentheses indicate a OpNode):
    //
    //   (AddOp,AddOp,)
    //

    graph = ark::OpGraph(model);
    UNITTEST_EQ(graph.get_nodes().size(), 1UL);

    node = graph.get_nodes().front().get();

    UNITTEST_EQ(node->ops[0]->outputs[0], t2);
    UNITTEST_EQ(node->ops[0]->inputs[0], t0);
    UNITTEST_EQ(node->ops[0]->inputs[1], t1);
    UNITTEST_EQ(node->ops[1]->outputs[0], t3);
    UNITTEST_EQ(node->ops[1]->inputs[0], t2);
    UNITTEST_EQ(node->ops[1]->inputs[1], t1);
    UNITTEST_EQ(node->users.size(), 0UL);
    UNITTEST_EQ(node->producers.size(), 0UL);

    // Test a chain of Ops without shared input tensors.
    // Model graph (omit leftmost part):
    //
    // ... ----+--> AddOp --> t3 ----+-> ReluOp --> t4
    // ...     |                     |
    // ... ----+   TensorOp --> tz --+
    // ...     |
    // ...   --+   (tz is the output reference, hidden from the code)
    //

    ark::Tensor *t4 = model.relu(t3);
    UNITTEST_TRUE(model.verify());

    // OpNode graph (parentheses indicate a OpNode):
    //
    //   (AddOp,AddOp,ReluOp,)
    //

    graph = ark::OpGraph(model);
    UNITTEST_EQ(graph.get_nodes().size(), 1UL);

    node = graph.get_nodes().front().get();

    UNITTEST_EQ(node->ops[0]->outputs[0], t2);
    UNITTEST_EQ(node->ops[0]->inputs[0], t0);
    UNITTEST_EQ(node->ops[0]->inputs[1], t1);
    UNITTEST_EQ(node->ops[1]->outputs[0], t3);
    UNITTEST_EQ(node->ops[1]->inputs[0], t2);
    UNITTEST_EQ(node->ops[1]->inputs[1], t1);
    UNITTEST_EQ(node->ops[2]->outputs[0], t4);
    UNITTEST_EQ(node->ops[2]->inputs[0], t3);
    UNITTEST_EQ(node->users.size(), 0UL);
    UNITTEST_EQ(node->producers.size(), 0UL);

    // Test a chain of Ops that use the output from the same previous Op.
    // Model graph (omit leftmost part):
    //
    // ...   +---- (this is t2) -------------------------+--> AddOp --> t5
    // ...   |                                           |
    // ... --+-+--> AddOp --> t3 ----+-> ReluOp --> t4 --+
    // ...     |                     |                   |
    // ... ----+   TensorOp --> tz --+                   |
    // ...     |                       TensorOp --> tw --+
    // ...   --+
    //
    // (tz and tw are output references, hidden from the code)
    //

    ark::Tensor *t5 = model.add(t2, t4);
    UNITTEST_TRUE(model.verify());

    // OpNode graph (parentheses indicate a OpNode):
    //
    //              +----------------------+
    //              |                      |
    //   (AddOp,) --+--> (AddOp,ReluOp,) --+--> (AddOp,)
    //

    graph = ark::OpGraph(model);
    UNITTEST_EQ(graph.get_nodes().size(), 3UL);

    auto nodes_iter = graph.get_nodes().begin();
    node = (nodes_iter++)->get();
    UNITTEST_EQ(node->ops[0]->outputs[0], t2);
    node = (nodes_iter++)->get();
    UNITTEST_EQ(node->ops[0]->outputs[0], t3);
    UNITTEST_EQ(node->ops[1]->outputs[0], t4);
    node = (nodes_iter++)->get();
    UNITTEST_EQ(node->ops[0]->outputs[0], t5);

    // Test an Op that uses outputs from multiple previous Ops.
    // Model graph (omit leftmost part):
    //
    // ... ----- (this is t2) --+--> AddOp --> t5
    // ...                      |              |
    // ... -+-> ReluOp --> t4 --+              |
    // ...  |                   |              |
    // ... -+                   |              |
    // ...    TensorOp --> tw --+              |
    // ...                                     |
    //                                         |
    //   TensorOp --> t6 --+--> AddOp --> t8 --+--> AddOp --> t9
    //                     |
    //   TensorOp --> t7 --+
    //                     |
    //   TensorOp --> tu --+
    //
    // (tw and tu are output references, hidden from the code)
    //

    ark::Tensor *t6 = model.tensor({1}, ark::FP32);
    ark::Tensor *t7 = model.tensor({1}, ark::FP32);
    ark::Tensor *t8 = model.add(t6, t7);
    ark::Tensor *t9 = model.add(t5, t8);
    UNITTEST_TRUE(model.verify());

    // OpNode graph (parentheses indicate a OpNode):
    //
    //              +----------------------+
    //              |                      |
    //   (AddOp,) --+--> (AddOp,ReluOp,) --+--> (AddOp,) --+
    //                                                     |
    //                                          (AddOp,) --+--> (AddOp,)
    //

    graph = ark::OpGraph(model);
    UNITTEST_EQ(graph.get_nodes().size(), 5UL);

    nodes_iter = graph.get_nodes().begin();
    node = (nodes_iter++)->get();
    UNITTEST_EQ(node->ops[0]->outputs[0], t2);
    node = (nodes_iter++)->get();
    UNITTEST_EQ(node->ops[0]->outputs[0], t3);
    UNITTEST_EQ(node->ops[1]->outputs[0], t4);
    node = (nodes_iter++)->get();
    UNITTEST_EQ(node->ops[0]->outputs[0], t5);
    node = (nodes_iter++)->get();
    UNITTEST_EQ(node->ops[0]->outputs[0], t8);
    node = (nodes_iter++)->get();
    UNITTEST_EQ(node->ops[0]->outputs[0], t9);

    // Test an Op that uses a single input tensor for multiple inputs.
    // Model graph (omit leftmost part):
    //
    // ... ----- (this is t2) --+--> AddOp --> t5
    // ...                      |              |
    // ... -+-> ReluOp --> t4 --+              |
    // ...  |                   |              |
    // ... -+                   |              |
    // ...    TensorOp --> tw --+              |
    // ...                                     |
    //                                         |
    //   TensorOp --> t6 --+--> AddOp --> t8 --+--> AddOp --> t9
    //                     |
    //   TensorOp --> t7 --+
    //                     |
    //   TensorOp --> tu --+
    //
    //   TensorOp --> t10 --+--> AddOp --> t11
    //                      |    ^  ^
    //                      |    |  |
    //                      +----+  |
    //                              |
    //   TensorOp --> tv -----------+
    //
    // (tw, tu, and tv are output references, hidden from the code)
    //

    ark::Tensor *t10 = model.tensor({1}, ark::FP32);
    ark::Tensor *t11 = model.add(t10, t10);
    UNITTEST_TRUE(model.verify());

    // OpNode graph (parentheses indicate a OpNode):
    //
    //              +----------------------+
    //              |                      |
    //   (AddOp,) --+--> (AddOp,ReluOp,) --+--> (AddOp,) --+
    //                                                     |
    //                                          (AddOp,) --+--> (AddOp,)
    //
    //                                                          (AddOp,)
    //

    graph = ark::OpGraph(model);
    UNITTEST_EQ(graph.get_nodes().size(), 6UL);

    nodes_iter = graph.get_nodes().begin();
    node = (nodes_iter++)->get();
    UNITTEST_EQ(node->ops[0]->outputs[0], t2);
    node = (nodes_iter++)->get();
    UNITTEST_EQ(node->ops[0]->outputs[0], t3);
    UNITTEST_EQ(node->ops[1]->outputs[0], t4);
    node = (nodes_iter++)->get();
    UNITTEST_EQ(node->ops[0]->outputs[0], t5);
    node = (nodes_iter++)->get();
    UNITTEST_EQ(node->ops[0]->outputs[0], t8);
    node = (nodes_iter++)->get();
    UNITTEST_EQ(node->ops[0]->outputs[0], t9);
    node = (nodes_iter++)->get();
    UNITTEST_EQ(node->ops[0]->outputs[0], t11);

    // Test using previous Ops' outputs from multiple different Ops.
    // Model graph (omit leftmost part):
    //
    // ... ----- (this is t2) --+--> AddOp --> t5
    // ...                      |              |
    // ... -+-> ReluOp --> t4 --+              |
    // ...  |                   |              |
    // ... -+                   |              |
    // ...    TensorOp --> tw --+              |
    // ...                                     |
    //                                         |
    //   TensorOp --> t6 --+--> AddOp --> t8 --+--> AddOp --> t9
    //                     |                   |
    //   TensorOp --> t7 --+                   +--> AddOp --> t12
    //                     |
    //   TensorOp --> tu --+
    //
    //   TensorOp --> t10 --+--> AddOp --> t11
    //                      |    ^  ^
    //                      |    |  |
    //                      +----+  |
    //                              |
    //   TensorOp --> tv -----------+
    //
    // (tw, tu, and tv are output references, hidden from the code)
    //

    ark::Tensor *t12 = model.add(t5, t8);
    UNITTEST_TRUE(model.verify());

    // OpNode graph (parentheses indicate a OpNode):
    //
    //              +----------------------+
    //              |                      |
    //   (AddOp,) --+--> (AddOp,ReluOp,) --+--> (AddOp,) --+--> (AddOp,)
    //                                                     |
    //                                          (AddOp,) --+--> (AddOp,)
    //
    //                                                          (AddOp,)
    //

    graph = ark::OpGraph(model);
    UNITTEST_EQ(graph.get_nodes().size(), 7UL);

    nodes_iter = graph.get_nodes().begin();
    node = (nodes_iter++)->get();
    UNITTEST_EQ(node->ops[0]->outputs[0], t2);
    node = (nodes_iter++)->get();
    UNITTEST_EQ(node->ops[0]->outputs[0], t3);
    UNITTEST_EQ(node->ops[1]->outputs[0], t4);
    node = (nodes_iter++)->get();
    UNITTEST_EQ(node->ops[0]->outputs[0], t5);
    node = (nodes_iter++)->get();
    UNITTEST_EQ(node->ops[0]->outputs[0], t8);
    node = (nodes_iter++)->get();
    UNITTEST_EQ(node->ops[0]->outputs[0], t9);
    node = (nodes_iter++)->get();
    UNITTEST_EQ(node->ops[0]->outputs[0], t11);
    node = (nodes_iter++)->get();
    UNITTEST_EQ(node->ops[0]->outputs[0], t12);

    return ark::unittest::SUCCESS;
}

ark::unittest::State test_sched_opgraph_noop()
{
    ark::Model model;
    model.tensor({1}, ark::FP32);
    model.tensor({1}, ark::FP32);
    model.tensor({1}, ark::FP32);
    UNITTEST_TRUE(model.verify());

    ark::OpGraph graph(model);
    UNITTEST_EQ(graph.get_nodes().size(), 0UL);
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_sched_opgraph_identity()
{
    // OpNode graph (parentheses indicate a OpNode):
    //
    //   (Relu,) --+
    //             |
    //   (Relu,) --+--> (Relu,)
    //

    ark::Model model;
    ark::Tensor *t0 = model.tensor({1}, ark::FP32);
    ark::Tensor *t1 = model.tensor({1}, ark::FP32);
    ark::Tensor *t2 = model.tensor({1}, ark::FP32);

    ark::Tensor *r0 = model.relu(t0);
    ark::Tensor *r1 = model.relu(t1);
    ark::Tensor *t3 = model.identity(t2, {r0, r1});

    ark::Tensor *t4 = model.relu(t3);
    UNITTEST_TRUE(model.verify());

    ark::OpGraph graph(model);
    UNITTEST_EQ(graph.get_nodes().size(), 3UL);

    auto nodes_iter = graph.get_nodes().begin();
    auto node = (nodes_iter++)->get();
    UNITTEST_EQ(node->ops[0]->outputs[0], r0);
    UNITTEST_EQ(node->producers.size(), 0UL);
    UNITTEST_EQ(node->users.size(), 1UL);

    node = (nodes_iter++)->get();
    UNITTEST_EQ(node->ops[0]->outputs[0], r1);
    UNITTEST_EQ(node->producers.size(), 0UL);
    UNITTEST_EQ(node->users.size(), 1UL);

    node = (nodes_iter++)->get();
    UNITTEST_EQ(node->ops[0]->outputs[0], t4);
    UNITTEST_EQ(node->producers.size(), 2UL);
    UNITTEST_EQ(node->users.size(), 0UL);

    return ark::unittest::SUCCESS;
}

ark::unittest::State test_sched_opgraph_sharding()
{
    // OpNode graph (parentheses indicate a OpNode):
    //
    //   (Relu,) --+
    //             |
    //   (Relu,) --+
    //             |
    //   (Relu,) --+--> (Relu,)
    //

    ark::Model model;
    ark::Tensor *t0 = model.tensor({3}, ark::FP32);

    std::vector<ark::Tensor *> vec = model.sharding(t0, 0, 1);
    UNITTEST_EQ(vec.size(), 3UL);

    ark::Tensor *t1 = vec[0];
    ark::Tensor *t2 = vec[1];
    ark::Tensor *t3 = vec[2];

    ark::Tensor *r0 = model.relu(t1);
    ark::Tensor *r1 = model.relu(t2);
    ark::Tensor *r2 = model.relu(t3);

    ark::Tensor *t4 = model.identity(t0, {r0, r1, r2});

    ark::Tensor *t5 = model.relu(t4);
    UNITTEST_TRUE(model.verify());

    ark::OpGraph graph(model);
    UNITTEST_EQ(graph.get_nodes().size(), 4UL);

    auto nodes_iter = graph.get_nodes().begin();
    auto node = (nodes_iter++)->get();
    UNITTEST_EQ(node->ops[0]->outputs[0], r0);
    UNITTEST_EQ(node->producers.size(), 0UL);
    UNITTEST_EQ(node->users.size(), 1UL);

    node = (nodes_iter++)->get();
    UNITTEST_EQ(node->ops[0]->outputs[0], r1);
    UNITTEST_EQ(node->producers.size(), 0UL);
    UNITTEST_EQ(node->users.size(), 1UL);

    node = (nodes_iter++)->get();
    UNITTEST_EQ(node->ops[0]->outputs[0], r2);
    UNITTEST_EQ(node->producers.size(), 0UL);
    UNITTEST_EQ(node->users.size(), 1UL);

    node = (nodes_iter++)->get();
    UNITTEST_EQ(node->ops[0]->outputs[0], t5);
    UNITTEST_EQ(node->producers.size(), 3UL);
    UNITTEST_EQ(node->users.size(), 0UL);

    return ark::unittest::SUCCESS;
}

ark::unittest::State test_sched_opgraph_split_matmul()
{
    // OpNode graph (parentheses indicate a OpNode):
    //
    //   (Matmul,) --+
    //               |
    //   (Matmul,) --+--> (Reduce,)
    //

    ark::Model model;
    ark::Tensor *t0 = model.tensor({64, 128}, ark::FP16);
    ark::Tensor *t1 = model.tensor({128, 64}, ark::FP16);
    ark::Tensor *m0 =
        model.matmul(t0, t1, nullptr, 2, false, false, false, "matmul", 3);
    UNITTEST_TRUE(model.verify());

    ark::OpGraph graph(model);
    UNITTEST_EQ(graph.get_nodes().size(), 3UL);

    auto nodes_iter = graph.get_nodes().begin();
    auto node = (nodes_iter++)->get();
    UNITTEST_EQ(node->producers.size(), 0UL);
    UNITTEST_EQ(node->users.size(), 1UL);

    node = (nodes_iter++)->get();
    UNITTEST_EQ(node->producers.size(), 0UL);
    UNITTEST_EQ(node->users.size(), 1UL);

    node = (nodes_iter++)->get();
    UNITTEST_EQ(node->ops[0]->outputs[0], m0);
    UNITTEST_EQ(node->producers.size(), 2UL);
    UNITTEST_EQ(node->users.size(), 0UL);

    return ark::unittest::SUCCESS;
}

int main()
{
    ark::init();
    UNITTEST(test_sched_opgraph);
    UNITTEST(test_sched_opgraph_noop);
    UNITTEST(test_sched_opgraph_identity);
    UNITTEST(test_sched_opgraph_sharding);
    UNITTEST(test_sched_opgraph_split_matmul);
    return 0;
}