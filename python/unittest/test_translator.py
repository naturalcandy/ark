import torch
from ark import ARKTranslator


def test_simple_model_translation():
    class SimpleModel(torch.nn.Module):
        def forward(self, x, y):
            a = torch.add(x, y)
            b = torch.mul(a, x)
            c = torch.mean(b)
            d = torch.nn.functional.relu(c)
            e = torch.reshape(d, (1,))
            f = torch.sqrt(e)
            return f

    model = SimpleModel()
    translator = ARKTranslator()
    ark_ir = translator.pytorch_to_ark(model)
    assert (
        len(ark_ir) == 9
    ), f"Expected 8 nodes, but got {len(ark_ir)}"  # 2 placeholders, 6 operations, 1 output

    operations = [
        node["Type"]
        for node in ark_ir
        if node["Type"] not in ["Placeholder", "Output"]
    ]
    expected_operations = [
        "add",
        "mul",
        "reduce_mean",
        "relu",
        "reshape",
        "sqrt",
    ]
    assert (
        operations == expected_operations
    ), f"Expected {expected_operations}, but got {operations}"

    add_node = next(node for node in ark_ir if node["Type"] == "add")
    assert (
        add_node["Args"]["arg_0"] == "%x"
    ), f"Expected arg_0 to be '%x', but got {add_node['Args']['arg_0']}"
    assert (
        add_node["Args"]["arg_1"] == "%y"
    ), f"Expected arg_1 to be '%y', but got {add_node['Args']['arg_1']}"


def test_unsupported_op():
    class UnsupportedModel(torch.nn.Module):
        def forward(self, x):
            return torch.erf(x)

    model = UnsupportedModel()
    translator = ARKTranslator()
    try:
        translator.pytorch_to_ark(model)
        assert False, "Expected ValueError, but no exception was raised"
    except ValueError as e:
        assert "Unsupported operation" in str(
            e
        ), f"Unexpected error message: {str(e)}"

test_unsupported_op()

test_simple_model_translation()