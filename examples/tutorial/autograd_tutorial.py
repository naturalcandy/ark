# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import ark
import torch
import torch.optim as optim
from torchviz import make_dot


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
        x = ark.matmul(x, self.weight_0, transpose_other=True)
        x = ark.matmul(x, self.weight_1, transpose_other=True)
        x = ark.matmul(x, self.weight_2, transpose_other=True)
        x = ark.matmul(x, self.weight_3, transpose_other=True)
        x = ark.matmul(x, self.weight_4, transpose_other=True)
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
    # Register the parameters with the RuntimeModule
    ark_module.register_parameter("weight_0", weight_0)
    ark_module.register_parameter("weight_1", weight_1)
    ark_module.register_parameter("weight_2", weight_2)
    ark_module.register_parameter("weight_3", weight_3)
    ark_module.register_parameter("weight_4", weight_4)

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


loss_fn = torch.nn.MSELoss()
optim_torch = optim.SGD(torch_model.parameters(), lr=0.01)
optim_ark = optim.SGD(ark_model.parameters(), lr=0.01)

num_iters = 5
for iter in range(num_iters):
    print(f"Iteration {iter+1}/{num_iters}")

    optim_torch.zero_grad()
    optim_ark.zero_grad()

    pytorch_output = torch_model(input_torch)
    with ark.torch_ctx.use_torch_autograd():
        ark_output = ark_model(input_ark)

    assert torch.allclose(pytorch_output, ark_output, atol=1e-4, rtol=1e-2)
    # for debugging
    pytorch_graph = make_dot(pytorch_output, params=dict(torch_model.named_parameters()))
    pytorch_graph.render(f"pytorch-autograd-graph-{iter+1}", format="png")

    # for debugging
    ark_graph = make_dot(ark_output, params=dict(ark_model.named_parameters()))
    ark_graph.render(f"ark-autograd-graph-{iter+1}", format="png")

    # Compute losses.
    torch_loss = loss_fn(pytorch_output, target)
    ark_loss = loss_fn(ark_output, target)

    # See how ARK's loss compares to PyTorch's loss.
    print(f"\nPyTorch loss: {torch_loss.item()}")
    print(f"\nARK loss: {ark_loss.item()}\n")
    assert torch.allclose(torch_loss, ark_loss, atol=1e-4, rtol=1e-2)

    # Perform a backward pass.
    torch_loss.backward()
    ark_loss.backward()

    optim_torch.step()
    optim_ark.step()

    # Ensure gradients of both models are updated accordingly.
    compare_grad(ark_model, torch_model)

