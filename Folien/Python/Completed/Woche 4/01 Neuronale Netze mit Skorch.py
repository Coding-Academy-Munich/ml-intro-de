# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>Neuronale Netze mit Skorch</b>
# </div>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# <br/>

# %% [markdown]
#
# # Training Neural Networks using Skorch
#
# Skorch is a library for PyTorch that simplifies training in a Scikit-Learn compatible way.

# %%
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import fetch_openml
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from skorch import NeuralNetClassifier

# %%
np.set_printoptions(precision=1)

# %%
mnist = globals().get("mnist") or fetch_openml("mnist_784", version=1)

# %% [markdown]
#
# ## Transforming the data:
#
# - Neural nets generally expect their inputs as `float32` (or possibly `float16`) values.
# - `skorch` expects classes to be stored as `int64` values.
# - We change the type of the arrays accordingly.

# %%
x = mnist.data.to_numpy().reshape(-1, 1, 28, 28).astype(np.float32)
y = mnist.target.to_numpy().astype(np.int64)
print("Shape of x:", x.shape, "- type of x:", x.dtype)
print("Shape of y:", y.shape, "- type of y:", y.dtype)

# %%
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=10_000, random_state=42
)
print("Shape of x train:", x_train.shape, "- shape of y_train:", y_train.shape)
print("Shape of x test:", x_test.shape, "- shape of y_test:", y_test.shape)

# %% [markdown]
#
# ## Normalize / Standardize
#
# - Neural networks generally prefer their input to be in the range $(0, 1)$ or $(-1, 1)$.
# - We need to convert the integer array to floating point.

# %%
print(x_train[0, 0, 20:24, 10:14])

# %% [markdown]
#
# - Don't use `StandardScaler` for this data, since it will scale each feature
#   independently.

# %%
scaler = StandardScaler()
x_train_scaled_with_scaler = scaler.fit_transform(x_train.reshape(-1, 28 * 28)).reshape(
    -1, 1, 28, 28
)
fig, axs = plt.subplots(ncols=2, figsize=(8, 4))
axs[0].imshow(x_train[0, 0], cmap="binary")
axs[1].imshow(x_train_scaled_with_scaler[0, 0], cmap="binary")

# %% [markdown]
#
# - Since we know the range of the values, we can easily perform the processing
#   manually.

# %%
x_train = x_train / 255.0
x_test = x_test / 255.0

# %%
print(x_train[0, 0, 20:24, 10:14])
plt.imshow(x_train[0, 0], cmap="binary")

# %% [markdown]
#
# ## Implementing the MLP Model using Basic PyTorch Functions
#
# - We can implement the model using basic PyTorch tensor operations, without
#   using `nn.Linear` or other high-level modules.
# - This approach demonstrates how neural network computations are built from
#   basic operations.


# %%
class MLPBasic(nn.Module):
    def __init__(self):
        super().__init__()
        # Manually initialize weights and biases
        self.W1 = nn.Parameter(torch.randn(28 * 28, 128) * 0.01)
        self.b1 = nn.Parameter(torch.zeros(128))
        self.W2 = nn.Parameter(torch.randn(128, 64) * 0.01)
        self.b2 = nn.Parameter(torch.zeros(64))
        self.W3 = nn.Parameter(torch.randn(64, 10) * 0.01)
        self.b3 = nn.Parameter(torch.zeros(10))

    def forward(self, x):
        # Flatten the input
        x = x.view(-1, 28 * 28)
        # First layer computations
        z1 = torch.matmul(x, self.W1) + self.b1
        a1 = torch.relu(z1)
        # Second layer computations
        z2 = torch.matmul(a1, self.W2) + self.b2
        a2 = torch.relu(z2)
        # Output layer computations
        z3 = torch.matmul(a2, self.W3) + self.b3
        # Do not apply Softmax here; CrossEntropyLoss does it internally
        return z3


# %%
mlp_basic_model = MLPBasic()

# %% [markdown]
#
# ## Implementing a Hand-Written Training Loop
#
# - Before using `NeuralNetClassifier`, we can train our model using a manual
#   training loop.
# - This demonstrates how backpropagation and optimization steps are performed.
# - We will use basic PyTorch functions to train the `MLPBasic` model.

# %%
import torch
from torch.utils.data import TensorDataset, DataLoader

# %%
# Convert numpy arrays to PyTorch tensors
x_train_tensor = torch.from_numpy(x_train)
y_train_tensor = torch.from_numpy(y_train)
x_test_tensor = torch.from_numpy(x_test)
y_test_tensor = torch.from_numpy(y_test)

# %%
# Create a TensorDataset and DataLoader for batching
train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# %%
# Initialize the model
model = MLPBasic()

# %%
# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# %%
# Number of epochs
n_epochs = 10

# %%
# Training loop
for epoch in range(n_epochs):
    running_loss = 0.0
    for inputs, labels in train_loader:
        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)

        # Compute loss
        loss = criterion(outputs, labels)

        # Backward pass (compute gradients)
        loss.backward()

        # Update weights
        optimizer.step()

        # Accumulate loss
        running_loss += loss.item()

    # Print statistics
    print(f"Epoch {epoch+1}/{n_epochs}, Loss: {running_loss/len(train_loader):.4f}")

# %%
# Evaluate on test set
with torch.no_grad():
    outputs = model(x_test_tensor)
    _, predicted = torch.max(outputs.data, 1)
    y_pred_manual = predicted.numpy()


# %%
# Print evaluation metrics
def print_scores(y, y_pred):
    print(f"Accuracy:          {accuracy_score(y, y_pred) * 100:.1f}%")
    print(f"Balanced accuracy: {balanced_accuracy_score(y, y_pred) * 100:.1f}%")
    print(
        f"Precision (macro): {precision_score(y, y_pred, average='macro') * 100:.1f}%"
    )
    print(f"Recall (macro):    {recall_score(y, y_pred, average='macro') * 100:.1f}%")
    print(f"F1 (macro):        {f1_score(y, y_pred, average='macro') * 100:.1f}%")


# %%
print_scores(y_test, y_pred_manual)

# %% [markdown]
#
# **Explanation:**
#
# - **Data Preparation:**
#   - Convert the NumPy arrays `x_train` and `y_train` to PyTorch tensors.
#   - Create a `TensorDataset` and `DataLoader` for batching the training data.
# - **Model Initialization:**
#   - Instantiate the `MLPBasic` model.
# - **Loss Function and Optimizer:**
#   - Use `nn.CrossEntropyLoss` as the loss function.
#   - Use Stochastic Gradient Descent (SGD) as the optimizer.
# - **Training Loop:**
#   - Iterate over the number of epochs.
#   - For each batch in the DataLoader:
#     - Zero the gradients using `optimizer.zero_grad()`.
#     - Perform the forward pass to compute model outputs.
#     - Compute the loss between the outputs and the target labels.
#     - Perform backpropagation using `loss.backward()`, which computes the gradients.
#     - Update the model parameters using `optimizer.step()`.
#     - Accumulate the running loss for monitoring.
#   - After each epoch, print the average loss.
# - **Evaluation:**
#   - Use `torch.no_grad()` to disable gradient computation during evaluation.
#   - Compute the outputs on the test set.
#   - Use `torch.max` to get the predicted classes.
#   - Convert predictions to NumPy array for evaluation.
# - **Results:**
#   - Use the `print_scores` function to display evaluation metrics.

# %% [markdown]
#
# ## Using `NeuralNetClassifier` for Training
#
# - `NeuralNetClassifier` is a Scikit-Learn compatible wrapper for PyTorch models.
# - It simplifies the training process by providing a high-level interface.
# - We can use it to train the `MLPBasic` model.
# - We can also move the training data to the GPU for faster computation.

# %%
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Training on {device}')

# %%
mlp_basic_model = MLPBasic()

# %%
mlp_basic_clf = NeuralNetClassifier(
    mlp_basic_model,
    criterion=nn.CrossEntropyLoss,
    max_epochs=10,
    lr=0.1,
    iterator_train__shuffle=True,
    device=device,
)

# %%
mlp_basic_clf.fit(x_train, y_train)

# %%
y_pred_mlp_basic = mlp_basic_clf.predict(x_test)

# %%
print_scores(y_test, y_pred_mlp_basic)

# %% [markdown]
#
# ## Implementing the MLP Model using `nn.Sequential`
#
# - `nn.Sequential` allows us to build models by specifying a sequence of
#   layers.
# - This is a simpler and more concise way to define models compared to manually
#   coding the computations.

# %%
mlp_model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28 * 28, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 10),
    # For CrossEntropyLoss, do not apply Softmax at the end
    # nn.Softmax(dim=1),
)

# %%
mlp_clf = NeuralNetClassifier(
    mlp_model,
    criterion=nn.CrossEntropyLoss,
    max_epochs=10,
    lr=0.1,
    iterator_train__shuffle=True,
    device=device,
)

# %%
mlp_clf.fit(x_train, y_train)

# %%
y_pred_mlp = mlp_clf.predict(x_test)

# %%
print_scores(y_test, y_pred_mlp)

# %% [markdown]
#
# ## Implementing the MLP Model using `nn.Module` (Object-Oriented Version)
#
# - By subclassing `nn.Module`, we can define models in an object-oriented way.
# - This approach provides greater flexibility, which is useful for more complex
#   models.


# %%
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        # Define layers
        self.flatten = nn.Flatten()
        self.hidden1 = nn.Linear(28 * 28, 128)
        self.relu1 = nn.ReLU()
        self.hidden2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.output_layer = nn.Linear(64, 10)
        # Note: We do not include Softmax here for CrossEntropyLoss

    def forward(self, x):
        x = self.flatten(x)
        x = self.hidden1(x)
        x = self.relu1(x)
        x = self.hidden2(x)
        x = self.relu2(x)
        x = self.output_layer(x)
        return x


# %%
mlp_model_oop = MLP()

# %%
mlp_clf_oop = NeuralNetClassifier(
    mlp_model_oop,
    criterion=nn.CrossEntropyLoss,
    max_epochs=10,
    lr=0.1,
    iterator_train__shuffle=True,
    device=device,
)

# %%
# mlp_clf_oop.fit(x_train, y_train)

# %%
# y_pred_mlp_oop = mlp_clf_oop.predict(x_test)

# %%
# print_scores(y_test, y_pred_mlp_oop)

# %% [markdown]
#
# ## Implementing a Convolutional Neural Network (CNN) using `nn.Sequential`
#
# - CNNs are well-suited for image data and can capture spatial hierarchies in
#   images.
# - Here, we define a simple CNN with two convolutional layers followed by fully
#   connected layers.

# %%
conv_model = nn.Sequential(
    nn.Conv2d(1, 10, kernel_size=5, stride=(2, 2)),
    nn.ReLU(),
    nn.Conv2d(10, 20, kernel_size=5, stride=(2, 2)),
    nn.ReLU(),
    nn.Flatten(),
    nn.Linear(320, 60),
    nn.ReLU(),
    nn.Linear(60, 10),
)

# %%
conv_clf = NeuralNetClassifier(
    conv_model,
    criterion=nn.CrossEntropyLoss,
    max_epochs=20,
    lr=0.1,
    iterator_train__shuffle=True,
    device=device,
)

# %%
# conv_clf.fit(x_train, y_train)

# %%
# y_pred_conv = conv_clf.predict(x_test)

# %%
# print_scores(y_test, y_pred_conv)

# %%
conv_model = nn.Sequential(
    nn.Conv2d(
        1, 32, kernel_size=3, padding=1
    ),  # Increased filters, smaller kernel, padding
    nn.BatchNorm2d(32),  # Added batch normalization
    nn.ReLU(),
    nn.Conv2d(32, 64, kernel_size=3, padding=1),  # Increased filters
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),  # Added MaxPooling
    nn.Conv2d(64, 128, kernel_size=3, padding=1),  # Added another conv layer
    nn.BatchNorm2d(128),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),  # Added MaxPooling
    nn.Flatten(),
    nn.Linear(128 * 7 * 7, 256),  # Adjusted input features based on conv output
    nn.ReLU(),
    nn.Dropout(0.5),  # Added dropout
    nn.Linear(256, 10),
)

# %%
from torch.optim.lr_scheduler import StepLR
from skorch.callbacks import LRScheduler

# %%
lr_scheduler = LRScheduler(
    policy=StepLR,         # The learning rate scheduler class
    step_size=5,           # Decay LR every 5 epochs
    gamma=0.2,             # Multiply LR by gamma
    verbose=True,          # Prints LR updates
)

# %%
# Define the classifier with the Adam optimizer and a lower learning rate
conv_clf = NeuralNetClassifier(
    conv_model,
    criterion=nn.CrossEntropyLoss,
    optimizer=torch.optim.Adam,       # Use the Adam optimizer
    max_epochs=20,
    lr=0.001,                         # Lower learning rate for Adam
    device=device,
    callbacks=[lr_scheduler],         # Add the LR scheduler callback
    iterator_train__shuffle=True,
    verbose=1,
)

# %%
# conv_clf.fit(x_train, y_train)

# %%
# y_pred_conv = conv_clf.predict(x_test)

# %%
# print_scores(y_test, y_pred_conv)

# %%
