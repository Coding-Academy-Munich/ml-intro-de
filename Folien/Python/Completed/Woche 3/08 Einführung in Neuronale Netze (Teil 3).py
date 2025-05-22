# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>Einführung in Neuronale Netze (Teil 3)</b>
# </div>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# <br/>

# %%
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from skorch import NeuralNetClassifier
from llm_utils import evaluate_model, plot_digits

# %% [markdown]
# ## Bessere Netzwerkarchitektur
#
# <img src="img/Figure-21-008.png" style="width: 30%; margin-left: auto; margin-right: auto; 0"/>

# %% [markdown]
# <img src="img/Figure-21-009.png" style="width: 40%; margin-left: auto; margin-right: auto; 0"/>

# %% [markdown]
# <img src="img/Figure-21-043.png" style="width: 40%; margin-left: auto; margin-right: auto; 0"/>

# %% [markdown]
# ## Beispiel: Conv Net
# %%
INPUT_SIZE = 28 * 28
NUM_CLASSES = 10
BATCH_SIZE = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# %%
mnist_transforms = transforms.Compose(
    [transforms.Resize((28, 28)), transforms.ToTensor()]
)

# %%
train_dataset = torchvision.datasets.MNIST(
    root="./localdata", train=True, transform=mnist_transforms, download=True
)

# %%
test_dataset = torchvision.datasets.MNIST(
    root="./localdata", train=False, transform=mnist_transforms, download=True
)

# %%
X_train = train_dataset.data.reshape(-1, 1, 28, 28) / 255.0
y_train = train_dataset.targets
X_test = test_dataset.data.reshape(-1, 1, 28, 28) / 255.0
y_test = test_dataset.targets

# %%
X_train.shape, y_train.shape, X_test.shape, y_test.shape


# %%
def create_conv_model():
    return nn.Sequential(
        nn.Conv2d(1, 32, kernel_size=3),
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=3),
        nn.MaxPool2d(2),
        nn.Dropout2d(0.5),
        nn.Flatten(1),
        nn.Linear(9216, 128),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(128, 10),
        nn.Softmax(dim=1),
    )


# %%
cnn = NeuralNetClassifier(
    create_conv_model,
    max_epochs=10,
    lr=0.002,
    optimizer=torch.optim.Adam,
    device=DEVICE,
)

# %%
cnn.fit(X_train, y_train)

# %%
evaluate_model(cnn, X_test, y_test)

# %%
y_pred = cnn.predict(X_test)

# %%
error_mask = y_test.numpy() != y_pred

# %%
plot_digits(X_test[error_mask], y_pred[error_mask], 5)

# %% [markdown]
#
# ## Workshop: Fashion MNIST
#
# Trainieren Sie ein Convolutional Neural Network auf dem Fashion MNIST
# Datensatz.
#
# *Hinweis:* Die Daten für den Fashion MNIST Datensatz sind im `torchvision`
# Paket enthalten. Sie können den Datensatz mit dem folgenden Code
# herunterladen:

# %%
fashion_train_dataset = torchvision.datasets.FashionMNIST(
    root="./localdata",
    train=True,
    transform=mnist_transforms,
    download=True,
)

# %%
fashion_test_dataset = torchvision.datasets.FashionMNIST(
    root="./localdata",
    train=False,
    transform=mnist_transforms,
    download=True,
)

# %%
X_train_fashion = fashion_train_dataset.data.reshape(-1, 1, 28, 28) / 255.0
y_train_fashion = fashion_train_dataset.targets
X_test_fashion = fashion_test_dataset.data.reshape(-1, 1, 28, 28) / 255.0
y_test_fashion = fashion_test_dataset.targets

# %%
fashion_cnn = NeuralNetClassifier(
    create_conv_model,
    max_epochs=10,
    lr=0.002,
    optimizer=torch.optim.Adam,
    device=DEVICE,
)

# %%
fashion_cnn.fit(X_train, y_train)

# %%
evaluate_model(cnn, X_test, y_test)
