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
# PyTorch is an open-source machine learning library developed by Facebook's AI
# Research lab. It provides:
#
# - **Tensors**: Similar to NumPy arrays, but with support for GPU acceleration.
# - **Dynamic Computation Graphs**: Allows flexibility in building and modifying
#   neural networks during runtime.
# - **Autograd Module**: Automatic differentiation for Tensors.
#
# In this notebook, we will explore PyTorch functionalities and learn how to
# build neural networks using PyTorch.

# %%
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from random import sample

# %%
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
# NumPy arrays but can be used for GPU acceleration and can automatically
# compute gradients, which are needed to train neural networks.
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
#
# The gradient of a function is the slope of the function at a given point. It
# is a generalization of the derivative of a function to multiple dimensions.
# PyTorch can automatically compute the gradient of a tensor with respect to
# some scalar value computed from the tensor.
#
# The `requires_grad` parameter specifies whether the tensor requires gradient
# computation. If `requires_grad=True`, PyTorch will track operations on the
# tensor. The gradient can be accessed using the `.grad` attribute.
#
# **Example**: Compute the gradient of $z = \sum_i x_i^2$ with respect to $x$.

# %%
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

# %%
x

# %%
y = x**2

# %%
z = y.sum()

# %% [markdown]
#
# The `backward()` method computes the gradient of the tensor with respect to
# some scalar value computed from the tensor. It is used to perform
# backpropagation in neural networks.
#
# PyTorch accumulates the gradient for all computations we perform on the
# tensor, even if we introduce intermediate variables (like `y` in this case).

# %%
z.backward()

# %% [markdown]
#
# Once we have called the `backward()` method, we can access the gradient of the
# tensor with respect to the scalar value computed from the tensor using the
# `.grad` attribute.

# %%
x.grad

# %% [markdown]
#
# The gradient of $z = \sum_i x_i^2$ with respect to $x_i$ is $2x_i$. Therefore,
# the gradient of $z$ with respect to $x$ is $[2, 4, 6]$.
# That is exactly the result we obtained above.

# %% [markdown]
#
# We can also compute the gradient of tensors with higher ranks:

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

# %% [markdown]
#
# The gradient of $z = \sum_{i,j} 2x_{ij}^3$ with respect to $x_{ij}$ is
# $2\cdot 3x_{ij}^2$. Therefore, the gradient of $z = [[1, 2, 3], [4, 5, 6]]$ with
# respect to $x$ is $[[6, 24, 54], [96, 150, 216]]$.

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

# %% [markdown]
#
# The `dim` parameter specifies the dimension along which the maximum values are
# computed. If `dim=0`, the maximum values are computed along the rows. If
# `dim=1`, the maximum values are computed along the columns.
#
# The function returns a tuple containing the maximum values and the indices of
# the maximum values along the specified dimension.

# %%
max_values, indices = torch.max(x, dim=0)
print("Max values along dim=1:", max_values)
print("Indices:", indices)

# %%
max_values, indices = torch.max(x, dim=1)
print("Max values along dim=1:", max_values)
print("Indices:", indices)

# %% [markdown]
#
# We can also pass two tensors with the same shape to the `torch.max` function.
# The function will return the maximum values and indices from the two tensors
# for each element.

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

# %% [markdown]
#
# ### Example: Simple Neural Network
#
# Let's build a simple neural network using PyTorch's `nn.Module` class.
#
# The neural network will have the following architecture:
#
# - Input layer: 2 neurons
# - Hidden layer: 3 neurons
# - Output layer: 1 neuron
#
# Since we use matrix multiplication to compute the output of each layer, the
# number of columns in the input must match the number of rows in the weight
# matrix. The weight matrix is defined as a tensor with shape `(input_size,
# output_size)`. The bias vector is defined as a tensor with shape
# `(output_size,)`. The output of a layer is computed as follows: $$z = xW + b$$
#
# Therefore we have
#
# - `W1` with shape `(2, 3)` mapping the 2 input neurons to the 3 hidden
#   neurons.
# - `b1` with shape `(3,)` adding the bias to the hidden layer.
# - `W2` with shape `(3, 1)` mapping the 3 hidden neurons to the 1 output
#   neuron.
# - `b2` with shape `(1,)` adding the bias to the output layer.
#
# The network will use the ReLU activation function for the hidden layer and
# will output logits for the output layer. The ReLU activation function takes
# the outputs of a layer and replaces all negative values with 0. Logits are
# simply the raw scores output by the network. In some cases another activation
# function (e.g., the sigmoid function) is applied to the logits to obtain
# values in the range [0, 1], but we will not do this here.
#
# The network will be defined as a class that inherits from `nn.Module`. The
# `__init__` method initializes the network's parameters, and the `forward`
# method defines the forward pass of the network, i.e., how the input is
# transformed into the output.
#
# The layers of the network will be defined as tensors wrapped `nn.Parameter`
# objects. By wrapping the tensors in `nn.Parameter` before assigning them to
# attributes of a `nn.Module`, PyTorch will automatically register them as
# parameters of the network. This allows PyTorch to track their gradients during
# backpropagation. We initialize the weights with small random values and the
# biases with zeros.

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
        a1 = torch.max(z1, torch.zeros_like(z1)) # ReLU activation
        z2 = a1 @ self.W2 + self.b2
        return z2  # Outputs logits


# %% [markdown]
#
# Let's create an instance of the `SimpleNet` class and pass some input data
# through the network.
#
# We set the random seed to ensure reproducibility.

# %%
torch.random.manual_seed(42)
simple_net = SimpleNet()

# %%
simple_net

# %% [markdown]
#
# The `forward` method of the network defines the forward pass of the network,
# i.e., how the input is transformed into the output. The `forward` method is
# called when we pass input data to the network. In the simplest case, we pass a
# tensor of shape `(input_size,)` to the network, where `input_size` is the
# number of input neurons, in our case 2.
#
# The `forward` method then computes the output of the network by applying the
# weights and biases of the network to the input data. The output of the network
# is the logits, i.e., the raw scores output by the network.
#
# Of course, the output of the network represents a random function, as we have
# initialized the weights randomly and not yet trained the network.

# %%
simple_net.forward(torch.ones(2))

# %% [markdown]
#
# Typically, we don't call the `forward` method directly. Instead, we use the
# network as a function, which internally calls the `forward` method.

# %%
simple_net(torch.ones(2))

# %% [markdown]
#
# Typically we don't want to compute the output of the network for a single
# input. Instead, we want to pass a batch of inputs to the network.
#
# This is particularly important when training the network, as we want to update
# the weights based on the gradients computed from multiple inputs. Therefore,
# the input to the network is usually a tensor of shape `(batch_size,
# input_size)`, where `batch_size` is the number of inputs in the batch.
#
# The Broadcasting rules in PyTorch allow us to pass a tensor of this shape to
# the network. In that case, we get a tensor of shape `(batch_size,
# output_size)` as the output.

# %%
simple_net(torch.tensor([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]]))

# %% [markdown]
#
# The network accumulates the gradients of the parameters during the forward
# pass. Typically we provide our training data multiple times to the network,
# since a single pass through the training data is not sufficient to learn the
# parameters of the network. Each pass through the training data is called an
# epoch.
#
# After each epoch, we update the parameters of the network based on the
# gradients computed during the forward pass. This process is called
# backpropagation.
#
# After each epoch we need to reset the gradients of the parameters to zero,
# since PyTorch accumulates the gradients by default. We can do this by calling
# the `zero_grad` method of the network.

# %%
simple_net.zero_grad()

# %% [markdown]
#
# ### Training a Simple Neural Network
#
# Let's train the simple neural network we defined above. We will generate some
# training data and define a loss function to measure the difference between the
# output of the network and the expected output.
#
# The training data will be a tensor of shape `(n_samples, 2)`, where
# `n_samples` is the number of samples in the training data. The expected output
# will be a tensor of shape `(n_samples, 1)`.
#
# The `generate_data` function generates random training data and expected
# output for the network. The expected output is the mean of the input data
# along the second dimension.


# %%
def generate_data(n_samples=5) -> tuple[torch.Tensor, torch.Tensor]:
    X = torch.rand((n_samples, 2), dtype=torch.float32) * 2 - 1
    y = X.mean(dim=1).reshape(-1, 1)
    n_samples = X.shape[0]
    return X, y


# %%
generate_data(3)

# %%
X_train, y_train = generate_data(500)

# %%
X_train[:5], y_train[:5]

# %% [markdown]
#
# If we apply our network to the training data, we expect a very bad
# performance, as the network has not been trained yet.
#
# How can we measure the performance of the network in a way that we can then
# use to improve its performance?
#
# There are many ways to measure the performance of a network. One common way is
# to use the mean absolute error (MAE) as the loss function. The MAE is the mean
# of the absolute differences between the predicted and true values.
#
# The MAE is defined as: $$\text{MAE} = \frac{1}{n} \sum_{i=1}^n |y_i -
# \hat{y}_i|$$
#
# where $y_i$ is the true value and $\hat{y}_i$ is the predicted value for the
# $i$-th sample.
#
# To compute the MAE, for our network, we run the network on input for which we
# know the true output. We then compute the absolute difference between the
# predicted and true output and take the mean of these differences.

# %%
result = simple_net(X_train)

# %%
result[:5]

# %%
loss = (result - y_train).abs().mean()

# %%
loss

# %% [markdown]
#
# The loss is very high, as expected, since the network has not been trained
# yet.
#
# To train the network, we need to update the parameters of the network based on
# the gradients computed during the forward pass. We can do this by calling the
# `backward` method of the loss tensor. The `backward` method computes the
# gradients of the loss with respect to the parameters of the network.
#
# After computing the gradients, we can update the parameters of the network
# using an optimization algorithm. For now, we will update the parameters
# manually using a learning rate of 0.1, i.e., we will subtract the gradient
# times the learning rate from the parameters.

# %%
loss.backward()

# %%
simple_net.W1.grad

# %%
simple_net.b1.grad

# %%
with torch.no_grad():
    for param in simple_net.parameters():
        param -= 0.1 * param.grad
    print(simple_net(X_train)[:5])

# %% [markdown]
#
# The output of the network has changed after updating the parameters. We can
# now compute the loss again and check if it has decreased.

# %%
(simple_net(X_train) - y_train).abs().mean()

# %% [markdown]
#
# The loss has decreased, which is a good sign, although the decrease was
# perhaps not as large as we would have liked, even though we used 500 samples.
#
# This shows one of the challenges of training neural networks: we need quite a
# lot of data to train the network effectively. In practice, we typically use
# thousands or even millions of samples to train a neural network.
#
# We can repeat the process we just applied multiple times to further decrease
# the loss: Send data through the network, adjust the parameters to (hopefully)
# reduce the loss, validate that we have achieved an improvement. This process
# is called training the network.
#
# We typically train the network for multiple epochs, i.e., we pass the training
# data to the network multiple times. After each epoch, we update the parameters
# of the network based on the gradients computed during the forward pass.
#
# It is important to reset the gradients of the parameters to zero after each
# epoch, as PyTorch accumulates the gradients by default and we don't want to
# accumulate the gradients from multiple epochs.


# %%
def run_training_loop(model, num_epochs=100, num_samples=5000, learning_rate=1e-4):
    X, y = generate_data(num_samples)
    for epoch in range(1, num_epochs + 1):
        result = model(X)
        loss = (result - y).abs().mean()
        loss.backward()
        with torch.no_grad():
            for param in model.parameters():
                param -= learning_rate * param.grad
        model.zero_grad()
        if epoch % (num_epochs / 10) == 0:
            print(f"Epoch {epoch}, loss: {loss.item():.5f}")


# %% [markdown]
#
# The function `print_results` generates new test data and prints the expected
# output, the predicted output, and the error for the first five samples. It
# also prints the mean error over all samples.

# %%
def print_results():
    X, y = generate_data(500)
    predicted = simple_net(X).detach()
    error = np.abs(predicted - y)
    values = (
        np.concatenate(
            [
                y,
                predicted,
                error,
            ],
            axis=1,
        )
    )
    print("Expected | Predicted | Error")
    print(values[:5])
    print(f"Mean error: {error.mean():.5f}")


# %%
print_results()

# %% [markdown]
#
# Let's train the network for 400 epochs and check the results.

# %%
torch.random.manual_seed(42)
simple_net = SimpleNet()

run_training_loop(simple_net, num_epochs=400, learning_rate=0.15)
print_results()

# %% [markdown]
#
# The mean error has decreased significantly after training the network for 400
# epochs. The network has learned to predict the mean of the input data much
# better than tha untrained net.
#
# We see that the loss during each training epoch decreased noticably. This is a
# good sign that the network is learning.
#
# However, if you try to train the network for many more epochs with the
# training rate of 0.15, you will notice that it does not become better, and
# even starts to get worse. This is because the learning rate, while initially
# successful in improving the net, is becoming too high, and the network starts
# to overshoot the minimum of the loss function.

# %%
run_training_loop(simple_net, num_epochs=1000, learning_rate=0.15)
print_results()

# %% [markdown]
#
# To avoid this problem, we can reduce the learning rate. A smaller learning rate
# means that the network updates the parameters more slowly, which can help to
# avoid overshooting the minimum of the loss function.
#
# Often we start the training with a relatively high learning rate and then
# decrease it during training. This is called learning rate scheduling.
#
# Let's try a simple learning rate scheduling: We start with a learning rate of
# 0.15 for 400 epochs, then decrease it to 0.1 for 250 epochs, then to 0.03 for
# 100 epochs, and so on. I've found these values by trial and error, but there
# are more sophisticated methods to find good learning rates.

# %%
torch.random.manual_seed(42)
simple_net = SimpleNet()

run_training_loop(simple_net, num_epochs=400, learning_rate=0.15)
print_results()

# %%
run_training_loop(simple_net, num_epochs=200, learning_rate=0.03)
print_results()

# %%
run_training_loop(simple_net, num_epochs=100, learning_rate=0.01)
print_results()

# %%
run_training_loop(simple_net, num_epochs=200, learning_rate=3e-3)
print_results()

# %%
run_training_loop(simple_net, num_epochs=200, learning_rate=5e-4)
print_results()

# %%
run_training_loop(simple_net, num_epochs=200, learning_rate=1e-6)
print_results()

# %% [markdown]
#
# By introducing this simple learning rate schedule, we were able to train the
# network to predict the mean of the input data almost perfectly: The mean error
# of the final network on the test data is less than 1e-4, a rather satisfying
# result.

# %% [markdown]
#
# Let's now turn to another method of defining neural networks in PyTorch: the
# `torch.nn.Sequential` class, and let's also look at a few more PyTorch
# functions and modules.
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
logits = torch.linspace(-2, 2, steps=10)
probabilities = torch.softmax(logits, dim=0)
plt.bar(logits.numpy(), probabilities.numpy())
plt.title("Softmax Function")
plt.xlabel("Logits")
plt.ylabel("Probabilities")
plt.grid(True)
plt.show()

# %%
logits = torch.linspace(-2, 2, steps=10)
probabilities = torch.softmax(logits, dim=0)
cumulative_probabilities = torch.cumsum(probabilities, dim=0)
plt.bar(logits.numpy(), cumulative_probabilities.numpy())
plt.title("Cumulative Distribution Function")
plt.xlabel("Logits")
plt.ylabel("Cumulative Probabilities")
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
loss.backward()

# %%
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
