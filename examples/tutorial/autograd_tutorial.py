# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import ark
import torch
import torch.optim as optim


# For debugging purposes, will remove and rewrite into
# `model_test_tutorial.py`

# Set random seed for reproducibility.
torch.manual_seed(42)


# Define a PyTorch model.
class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(256, 256, bias=False),  # Layer 0
            torch.nn.Linear(256, 256, bias=False),  # Layer 1
            torch.nn.Linear(256, 256, bias=False),  # Layer 2
            torch.nn.Linear(256, 256, bias=False),  # Layer 3
            torch.nn.Linear(256, 256, bias=False),  # Layer 4
            torch.nn.ReLU(),  # Activation
        )

    def forward(self, x):
        return self.layers(x)


torch_model = SimpleModel()
torch_model.to("cuda:0")


# Define an ARK model with the equivalent architecture.
class SimpleModelARK(ark.Module):
    def __init__(self, weight_0, weight_1, weight_2, weight_3, weight_4):
        super().__init__()
        self.weight_0 = weight_0
        self.weight_1 = weight_1
        self.weight_2 = weight_2
        self.weight_3 = weight_3
        self.weight_4 = weight_4

    def forward(self, x):
        x = ark.matmul(x, self.weight_0)
        x = ark.matmul(x, self.weight_1)
        x = ark.matmul(x, self.weight_2)
        x = ark.matmul(x, self.weight_3)
        x = ark.matmul(x, self.weight_4)
        x = ark.relu(x)
        return x


def replace_layers_with_ark(model):
    weight_0 = torch.nn.Parameter(
        model.layers[0].weight.to("cuda:0").requires_grad_(True)
    )
    weight_1 = torch.nn.Parameter(
        model.layers[1].weight.to("cuda:0").requires_grad_(True)
    )
    weight_2 = torch.nn.Parameter(
        model.layers[2].weight.to("cuda:0").requires_grad_(True)
    )
    weight_3 = torch.nn.Parameter(
        model.layers[3].weight.to("cuda:0").requires_grad_(True)
    )
    weight_4 = torch.nn.Parameter(
        model.layers[4].weight.to("cuda:0").requires_grad_(True)
    )
    ark_module = ark.RuntimeModule(
        SimpleModelARK(weight_0, weight_1, weight_2, weight_3, weight_4)
    )
    return ark_module


ark_model = replace_layers_with_ark(torch_model)


input_torch = torch.randn(128, 256).to("cuda:0").requires_grad_(True)
input_ark = input_torch.clone().detach().requires_grad_(True)

target = torch.randn(128, 256).to("cuda:0")


# Function to compare the gradients of two models of the same architecture and parameter order.
def compare_grad(ark_model, torch_model, atol=1e-4, rtol=1e-2):
    ark_params = list(ark_model.named_parameters())
    torch_params = list(torch_model.named_parameters())
    for (ark_name, ark_param), (torch_name, torch_param) in zip(
        ark_params, torch_params
    ):
        if (ark_param.grad is None) ^ (torch_param.grad is None):
            print("Exactly one of the gradients is None")
        else:
            grads_equal = torch.allclose(
                ark_param.grad, torch_param.grad, atol=atol, rtol=rtol
            )
            if not grads_equal:
                print(
                    f"Gradient for {ark_name} when compared to {torch_name} is different:"
                )
                print(f"ARK gradient: {ark_param.grad}")
                print(f"Torch gradient: {torch_param.grad}")


torch_output = torch_model(input_torch)
with ark.torch_ctx.use_torch_autograd():
    ark_output = ark_model(input_ark)

# Compare outputs
assert torch.allclose(torch_output, ark_output, atol=1e-4, rtol=1e-2)
