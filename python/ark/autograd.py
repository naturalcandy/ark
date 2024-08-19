# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from .tensor import Tensor
from typing import List, Optional, Any, Callable

try:
    import torch

    _no_torch = False
except ImportError:
    from . import torch_mock as torch

    _no_torch = True


class _OperationRegistryImpl:
    """
    A bridge class for translating ARK operations to proxy PyTorch operations
    in order to construct a backwards graph using PyTorch's autograd.

    This class maintains a sequence of operations (op_seq) performed in ARK that need to be
    replicated in PyTorch. Each operation in the ops_list is represented by a tuple with
    the following structure:

    (
        [input_tensor_ids],   # List of input tensor IDs used in the op
        [output_tensor_ids],  # List of output tensor IDs produced from the op
        torch_operation,      # The corresponding PyTorch operation
        args,                 # Additional positional arguments
        kwargs                # Additional keyword arguments
    )

    The class provides methods to record the operations performed in ARK,
    and to simulate them later on, using PyTorch's API.
    """

    def __init__(self):
        self.handled_functions = {}
        self.op_seq = []
        self.tns_map = {}
        self.op_idx = -1

    def implements(self, torch_function):
        def decorator(func):
            self.handled_functions[torch_function] = func
            return func

        return decorator

    def record_op(
        self,
        input_ids: List[int],
        output_ids: List[int],
        op: Callable,
        *args: Any,
        **kwargs: Any,
    ):
        self.op_seq.append((input_ids, output_ids, op, args, kwargs))

    def get_output(self):
        if self.op_idx >= 0 and self.op_idx < len(self.op_seq):
            output_ids = self.op_seq[self.op_idx][1]
            return [self.tns_map[id] for id in output_ids]
        raise RuntimeError(
            "No matching operation found or invalid operation index"
        )

    def clear(self):
        self.op_seq.clear()
        self.tns_map.clear()
        self.op_idx = -1

    def execute_overridden_ops(self):
        for i, (input_ids, output_ids, op, args, kwargs) in enumerate(
            self.op_seq
        ):
            if op in self.handled_functions:
                self.op_idx = i
                input_tensors = [self.tns_map[id] for id in input_ids]
                self.handled_functions[op](*input_tensors, *args, **kwargs)
            else:
                raise NotImplementedError(
                    f"ARK operation {op} not implemented in PyTorch"
                )
        self.op_idx = -1


_OperationRegistry = _OperationRegistryImpl()


# Overriden torch operations
@_OperationRegistry.implements(torch.matmul)
def matmul(input, other):
    outputs = _OperationRegistry.get_output()
    return outputs


@_OperationRegistry.implements(torch.nn.functional.relu)
def relu(input, inplace):
    outputs = _OperationRegistry.get_output()
    return outputs
