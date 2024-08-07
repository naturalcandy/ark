from ark import ops as ark_ops
import torch
import torch.fx as fx
import torch.nn.functional as F

from typing import List


class ARKTranslator:
    def __init__(self):
        self.op_map = {
            # Basic arithmetic operations
            torch.add: ark_ops.add,
            torch.sub: ark_ops.sub,
            torch.mul: ark_ops.mul,
            torch.div: ark_ops.div,
            # Activation functions
            torch.relu: ark_ops.relu,
            F.relu: ark_ops.relu,
            torch.sigmoid: ark_ops.sigmoid,
            F.sigmoid: ark_ops.sigmoid,
            torch.exp: ark_ops.exp,
            F.gelu: ark_ops.gelu,
            # Matrix operations
            torch.matmul: ark_ops.matmul,
            torch.mm: ark_ops.matmul,
            # Reduction operations
            torch.max: ark_ops.reduce_max,
            torch.mean: ark_ops.reduce_mean,
            torch.sum: ark_ops.reduce_sum,
            # Shape operations
            torch.reshape: ark_ops.reshape,
            torch.transpose: ark_ops.transpose,
            # Other operations
            torch.sqrt: ark_ops.sqrt,
            torch.rsqrt: ark_ops.rsqrt,
            F.softmax: ark_ops.softmax,
            F.layer_norm: ark_ops.layernorm,
            torch.nn.functional.embedding: ark_ops.embedding,
            # Tensor creation
            torch.ones: ark_ops.ones,
            torch.zeros: ark_ops.zeros,
            torch.full: ark_ops.constant,
            # Special operations
            "rope": ark_ops.rope,
            "all_reduce": ark_ops.all_reduce,
            # Identity and no-op
            torch.nn.Identity: ark_ops.identity,
            "noop": ark_ops.noop,
        }

    def translate_node(self, node: fx.Node) -> dict:
        if node.op == "placeholder":
            return {"Type": "Placeholder", "Args": {"name": node.target}}
        elif node.op == "get_attr":
            return {"Type": "GetAttr", "Args": {"name": node.target}}
        elif node.op == "call_function":
            ark_op = self.op_map.get(node.target)
            if ark_op is None:
                raise ValueError(f"Unsupported operation: {node.target}")
            return {
                "Type": ark_op.__name__,
                "Args": self.translate_args(node.args, node.kwargs),
            }
        elif node.op == "output":
            return {"Type": "Output", "Args": self.translate_args(node.args)}
        else:
            raise ValueError(f"Unsupported node op: {node.op}")

    def translate_args(self, args, kwargs=None):
        translated_args = {}
        for i, arg in enumerate(args):
            if isinstance(arg, fx.Node):
                translated_args[f"arg_{i}"] = f"%{arg.name}"
            else:
                translated_args[f"arg_{i}"] = arg
        if kwargs:
            for key, value in kwargs.items():
                if isinstance(value, fx.Node):
                    translated_args[key] = f"%{value.name}"
                else:
                    translated_args[key] = value
        return translated_args

    def translate_graph(self, graph: fx.Graph) -> List[dict]:
        ark_ir = []
        for node in graph.nodes:
            ark_node = self.translate_node(node)
            ark_ir.append(ark_node)
        return ark_ir

    def pytorch_to_ark(self, model: torch.nn.Module) -> List[dict]:
        graph_module = fx.symbolic_trace(model)
        return self.translate_graph(graph_module.graph)
