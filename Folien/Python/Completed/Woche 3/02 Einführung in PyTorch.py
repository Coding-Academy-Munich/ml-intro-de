# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>Einführung in PyTorch</b>
# </div>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# <br/>

# %% [markdown]
#
# # Introduction to PyTorch
#
# PyTorch is an open-source machine learning library developed by Facebook's AI Research lab. It provides:
#
# - **Tensors**: Similar to NumPy arrays, but with support for GPU acceleration.
# - **Dynamic Computation Graphs**: Allows flexibility in building and modifying neural networks during runtime.
# - **Autograd Module**: Automatic differentiation for Tensors.
#
# In this notebook, we will explore PyTorch functionalities and learn how to build neural networks using PyTorch.

# %%
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from skorch import NeuralNetClassifier

# %% [markdown]
#
# ## PyTorch Basics
#
# ### Tensors
#
# Tensors are the fundamental data structure in PyTorch. They are similar to
# NumPy arrays but can be used for GPU acceleration.
#
# - Parameters: `data`, `dtype`, `device`, `requires_grad`.
# - `data` can be a list, tuple, NumPy array, scalar, etc.
# - `dtype` specifies the data type of the tensor.
# - `device` specifies the device (CPU or GPU) where the tensor will be stored.
# - `requires_grad` specifies whether the tensor requires gradient computation.

# %%
a = torch.tensor([1, 2, 3])

# %%
a

# %%
type(a)

# %%
b = torch.tensor([4, 5, 6])

# %%
b

# %%
a + b

# %%
a * b

# %% [markdown]
#
# #### Example for Gradient Computation

# %%
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

# %%
x

# %%
y = x**2

# %%
z = y.sum()

# %%
z.backward()

# %%
x.grad


# %%
mat = np.linspace(1.0, 6.0, 6).reshape(2, 3)

# %%
x = torch.tensor(mat, requires_grad=True)

# %%
x

# %%
y = 2 * x**3

# %%
z = y.sum()

# %%
z.backward()

# %%
x.grad

# %%
2 * 3 * mat**2


# %% [markdown]
#
# ### Converting between NumPy arrays and PyTorch tensors

# %%
a_np = a.numpy()

# %%
a_np

# %%
type(a_np)

# %%
np_array = np.array([7, 8, 9])

# %%
tensor_from_np = torch.from_numpy(np_array)

# %%
tensor_from_np

# %%
type(tensor_from_np)

# %% [markdown]
#
# ## PyTorch Functions and Modules
#
# Let's explore some PyTorch functions used in building neural networks.

# %% [markdown]
#
# ### `torch.randn`
#
# **Description**:
#
# - Returns a tensor filled with random numbers from a normal distribution.
# - Parameters: `size`, `dtype`, `device`, `requires_grad`.
# - `size` can be a single integer, a tuple of integers, or passed as several
#   parameters.
# - `dtype` specifies the data type of the output tensor.
# - `device` specifies the device (CPU or GPU) where the tensor will be stored.
# - `requires_grad` specifies whether the tensor requires gradient computation.
#   - Gradients are used for backpropagation in neural networks.
#   - If `requires_grad=True`, PyTorch will track operations on the tensor.
#   - The gradient can be accessed using the `.grad` attribute.

# %%
torch.randn(2, 3)

# %%
torch.randn((2, 3))

# %%
torch.randn(2, 3, dtype=torch.float16)

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
torch.randn(2, 3, device=device)

# %%
torch.randn(2, 3, requires_grad=True)

# %% [markdown]
#
# ### torch.view()
#
# **Description**:
#
# - Returns a new tensor with the same data but a different shape.
# - The returned tensor shares the same data with the original tensor.

# %%
x = torch.randn(2, 3, 4)

# %%
x.shape

# %%
x_view = x.view(-1, 4)  # Flatten the first two dimensions

# %%
x_view.shape

# %% [markdown]
#
# ### torch.max
#
# **Description**:
#
# - Returns the maximum value of all elements in the input tensor.

# %%
x = torch.tensor([1.0, 3.0, 2.0, 5.0])

# %%
torch.max(x)

# %%
x = torch.tensor([[1, 2], [3, 4], [4, 2]])

# %%
max_values, indices = torch.max(x, dim=0)
print("Max values along dim=1:", max_values)
print("Indices:", indices)

# %%
max_values, indices = torch.max(x, dim=1)
print("Max values along dim=1:", max_values)
print("Indices:", indices)

# %%
torch.max(torch.linspace(-3, 3, steps=7), torch.zeros(7))

# %% [markdown]
#
# ### torch.nn.Module
#
# **Description**:
#
# - Base class for all neural network modules.
# - Your models should subclass `nn.Module`.
# - Modules can contain other modules, allowing for nested structures.


# %%
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.W1 = nn.Parameter(torch.randn(2, 3) * 0.01)
        self.b1 = nn.Parameter(torch.zeros(3))
        self.W2 = nn.Parameter(torch.randn(3, 1) * 0.01)
        self.b2 = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        z1 = x @ self.W1 + self.b1
        a1 = torch.max(z1, torch.zeros_like(z1))
        z2 = a1 @ self.W2 + self.b2
        return z2  # Outputs logits


# %%
simple_net = SimpleNet()

# %%
simple_net

# %%
simple_net.forward(torch.ones(2))

# %%
simple_net(torch.ones(2))

# %%
simple_net.zero_grad()


# %%
def generate_training_data(max=5, shuffle=True):
    values = torch.cat(
        [torch.tensor([[n, n + 2]], dtype=torch.float32) for n in range(-max, max + 1)],
        dim=0,
    )
    n_samples = values.shape[0]
    shuffle_indices = torch.randperm(n_samples) if shuffle else torch.arange(n_samples)
    return values[shuffle_indices]


# %%
generate_training_data(3, shuffle=False)

# %%
training_data = generate_training_data()

# %%
training_data


# %%
def results_from_training_data(training_data):
    return 0.01 * (training_data[:, 0].reshape(-1, 1) + 1)


# %%
expected_output = results_from_training_data(training_data)

# %%
expected_output

# %%
result = simple_net(training_data)

# %%
result

# %%
loss = (result - expected_output).abs().mean()

# %%
loss

# %%
loss.backward()

# %%
simple_net.W1.grad

# %%
simple_net.b1.grad

# %%
eval_data = generate_training_data(5, shuffle=False)
results_from_training_data(eval_data)

# %%
# Update the parameters
with torch.no_grad():
    for param in simple_net.parameters():
        param -= 0.1 * param.grad
    print(simple_net(eval_data))


# %%
def loss():
    return (simple_net(eval_data) - expected_output).abs().mean().item()


# %%
loss()


# %%
def run_training_loop(model, num_epochs=100, num_samples=2000, learning_rate=1e-4):
    training_data = generate_training_data(num_samples)
    expected_output = results_from_training_data(training_data)
    for epoch in range(num_epochs):
        result = model(training_data)
        loss = (result - expected_output).abs().mean()
        loss.backward()
        with torch.no_grad():
            for param in model.parameters():
                param -= learning_rate * param.grad
        model.zero_grad()
        if epoch % (num_epochs / 10) == 0:
            print(f"Epoch {epoch}, loss: {loss.item():.5f}")


# %%
def print_results():
    eval_data = generate_training_data(5, shuffle=False)
    expected = results_from_training_data(eval_data).detach().numpy()
    predicted = simple_net(eval_data).detach().numpy()
    error = np.abs(predicted - expected)
    values = (
        np.concatenate(
            [
                expected,
                predicted,
                error,
            ],
            axis=1,
        )
        * 10_000
    )
    np.set_printoptions(precision=2)
    print("Expected | Predicted | Error")
    print(values)


# %%
print_results()

# %%
torch.random.manual_seed(42)
simple_net = SimpleNet()

# %%
print_results()

# %%
run_training_loop(simple_net)

# %%
print_results()

# %%
torch.random.manual_seed(42)
simple_net = SimpleNet()
run_training_loop(simple_net, num_epochs=60)

# %%
print_results()

# %%
run_training_loop(simple_net, num_epochs=100, learning_rate=1e-5)

# %%
print_results()

# %%
run_training_loop(simple_net, num_epochs=1000, learning_rate=1e-6)

# %%
print_results()

# %% [markdown]
#
# ### `torch.nn.Flatten` and `torch.flatten`
#
# **Description**:
#
# - Flattens a contiguous range of dimensions into a tensor.
# - Commonly used to flatten inputs before feeding them into fully connected
#   layers.
# - `nn.Flatten` is a module, while `torch.flatten` is a functional interface.

# %%
flatten = nn.Flatten()

# %%
input_tensor = torch.randn(1, 2, 3)

# %%
input_tensor.shape

# %%
flattened_tensor = flatten(input_tensor)

# %%
flattened_tensor.shape

# %% [markdown]
#
# ### torch.nn.Linear
#
# **Description**:
#
# - Applies a linear transformation to the incoming data: $y = xA^T + b$
# - Parameters: `in_features`, `out_features`

# %%
linear_layer = nn.Linear(3, 2)
input_tensor = torch.randn(1, 3)
output_tensor = linear_layer(input_tensor)

# %%
input_tensor

# %%
output_tensor

# %%
linear_layer.weight

# %%
linear_layer.bias

# %%
input_tensor @ linear_layer.weight.T + linear_layer.bias

# %%
output_tensor

# %% [markdown]
#
# ### torch.nn.ReLU and torch.relu
#
# **Description**:
#
# - The ReLU activation function introduces non-linearity.
# - `nn.ReLU` is a module, while `torch.relu` is a functional interface.

# %%
x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])

# %%
x

# %%
torch.relu(x)


# %% [markdown]
#
# **Visualization**:

# %%
relu_module = nn.ReLU()
x = torch.linspace(-5, 5, steps=100)
y = relu_module(x)
plt.plot(x.numpy(), y.numpy())
plt.title("ReLU Activation Function")
plt.xlabel("Input")
plt.ylabel("Output")
plt.grid(True)
plt.show()

# %% [markdown]
#
# ### torch.nn.Softmax and torch.softmax
#
# **Description**:
#
# - Applies the Softmax function to a tensor.
# - Converts logits into probabilities that sum to 1.
#
# **Visualization**:

# %%
softmax_module = nn.Softmax(dim=0)
logits = torch.tensor([1.0, 2.0, 3.0])
probabilities = softmax_module(logits)
print("Logits:", logits)
print("Probabilities after Softmax:", probabilities)
print("Sum of probabilities:", probabilities.sum())

# %%
logits = torch.linspace(-2, 2, steps=100)
probabilities = torch.softmax(logits, dim=0)
plt.plot(logits.numpy(), probabilities.numpy())
plt.title("Softmax Function")
plt.xlabel("Logits")
plt.ylabel("Probabilities")
plt.grid(True)
plt.show()

# %% [markdown]
#
# ### torch.nn.Conv2d
#
# **Description**:
#
# - Applies a 2D convolution over an input signal composed of several input planes.

# %%
# Example of a Conv2d layer
conv_layer = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3)
input_image = torch.randn(1, 1, 5, 5)  # Batch size, channels, height, width
output_image = conv_layer(input_image)
print("Input shape:", input_image.shape)
print("Output shape:", output_image.shape)

# %% [markdown]
#
# ### torch.nn.Sequential
#
# **Description**:
#
# - A sequential container to build neural networks.
# - Modules will be added in the order they are passed.

# %%
# Example of using nn.Sequential
sequential_model = nn.Sequential(
    nn.Flatten(), nn.Linear(28 * 28, 128), nn.ReLU(), nn.Linear(128, 10)
)
print(sequential_model)

# %% [markdown]
#
# ### torch.nn.CrossEntropyLoss
#
# **Description**:
#
# - Combines `nn.LogSoftmax()` and `nn.NLLLoss()` in a single class.
# - Useful for multi-class classification problems.
#
# **Important Concept**:
#
# **Cross-Entropy Loss** measures the performance of a classification model
# whose output is a probability value between 0 and 1. It increases as the
# predicted probability diverges from the actual label.
#
# In the context of multi-class classification:
#
# - The model outputs a vector of raw scores (logits) for each class.
# - These scores are converted to probabilities using the softmax function.
# - The cross-entropy loss is then calculated between these probabilities and
#   the one-hot encoded true labels.

# %%
criterion = nn.CrossEntropyLoss()

# %%
logits = torch.tensor([[0.5, 1.5, 0.3, 0.6]])

# %%
target = torch.tensor([1])  # True class index

# %%
loss = criterion(logits, target)

# %%
loss.item()

# %%
sm = torch.softmax(logits.squeeze(), dim=0)

# %%
sm

# %%
torch.log(sm[target]).item()

# %% [markdown]
#
# Very confident correct prediction

# %%
logits = torch.tensor([[0.1, 10.0, 0.1, 0.1]])
target = torch.tensor([1])
loss = criterion(logits, target)
print(loss.item())

# %% [markdown]
#
# Very confident incorrect prediction

# %%
logits = torch.tensor([[0.1, 0.1, 10.0, 0.1]])
target = torch.tensor([1])
loss = criterion(logits, target)
print(loss.item())  # Output: 9.999954223632812


# %% [markdown]
#
# Uncertain prediction

# %%
logits = torch.tensor([[2.0, 2.1, 1.9, 2.05]])
target = torch.tensor([1])
loss = criterion(logits, target)
print(loss.item())

# %% [markdown]
#
# Multi-sample example

# %%
logits = torch.tensor([[1.0, 2.0, 0.5, 0.8]])
target = torch.tensor([1])
loss1 = criterion(logits, target)
print(loss1.item())

# %%
logits = torch.tensor([[0.2, 0.3, 0.9, 1.2]])
target = torch.tensor([3])
loss2 = criterion(logits, target)
print(loss2.item())

# %%
logits = torch.tensor([[2.0, 1.0, 3.0, 0.5]])
target = torch.tensor([2])
loss3 = criterion(logits, target)
print(loss3.item())

# %%
(loss1 + loss2 + loss3) / 3

# %%
logits = torch.tensor([[1.0, 2.0, 0.5, 0.8],
                       [0.2, 0.3, 0.9, 1.2],
                       [2.0, 1.0, 3.0, 0.5]])
target = torch.tensor([1, 3, 2])
loss = criterion(logits, target)
print(loss.item())



# %% [markdown]
#
# ### torch.optim.SGD
#
# **Description**:
#
# - Implements stochastic gradient descent optimization algorithm.
#
# **Example**:

# %%
w = torch.randn(2, 2, requires_grad=True)

# %%
optimizer = torch.optim.SGD([w], lr=0.01)

# %%
loss = (w**2).sum()

# %%
# Backpropagation
loss.backward()

# %%
# Update parameters
optimizer.step()
print("Updated parameters:", w)

# %% [markdown]
#
# ### torch.utils.data.TensorDataset and DataLoader
#
# **Description**:
#
# - `TensorDataset`: Wraps tensors to create a dataset.
# - `DataLoader`: Provides an iterable over the dataset.

# %%
x_tensor = torch.randn(100, 3)
y_tensor = torch.randint(0, 2, (100,))

# %%
dataset = TensorDataset(x_tensor, y_tensor)

# %%
dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

# %%
# Iterate over the DataLoader
for batch_idx, (data, target) in enumerate(dataloader):
    print(f"Batch {batch_idx+1}: Data shape {data.shape}, Target shape {target.shape}")
    break  # Just show one batch

# %% [markdown]
#
# ### torch.no_grad()
#
# **Description**:
#
# - Context-manager that disables gradient calculation.
# - Reduces memory consumption during evaluation.

# %%
x = torch.randn(5, requires_grad=True)

# %%
y = (x * 2).sum()

# %%
x, y

# %%
y.backward()

# %%
with torch.no_grad():
    y = (x * 2).sum()

# %%
x, y

# %%
# y.backward()  # This will raise an error
