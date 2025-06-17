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
from skorch import NeuralNetClassifier
from skorch.callbacks import LRScheduler
from llm_utils import evaluate_model, plot_digits

# %% [markdown]
#
# ## Bessere Netzwerkarchitektur

# %% [markdown]
# <img src="img/Figure-21-008.png" style="width: 30%; margin-left: auto; margin-right: auto;"/>

# %% [markdown]
# <img src="img/Figure-21-009.png" style="width: 40%; margin-left: auto; margin-right: auto;"/>

# %% [markdown]
# <img src="img/Figure-21-043.png" style="width: 40%; margin-left: auto; margin-right: auto;"/>

# %% [markdown]
#
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


# %% [markdown]
#
# ### Definieren des Modells

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


# %% [markdown]
#
# ### Erzeugen des Skorch-Classifiers

# %%
cnn = NeuralNetClassifier(
    create_conv_model,
    max_epochs=10,
    lr=0.002,
    optimizer=torch.optim.Adam,
    device=DEVICE,
)


# %% [markdown]
#
# ### Trainieren des Modells

# %%
cnn.fit(X_train, y_train)

# %% [markdown]
#
# ### Evaluieren des Modells

# %%
evaluate_model(cnn, X_test, y_test)

# %% [markdown]
#
# ### Analysieren der Fehler

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
fashion_cnn.fit(X_train_fashion, y_train_fashion)

# %%
y_pred_fashion = fashion_cnn.predict(X_test_fashion)

# %%
plot_digits(X_test_fashion, y_pred_fashion, 5)

# %%
error_mask_fashion = y_test_fashion.numpy() != y_pred_fashion

# %%
evaluate_model(fashion_cnn, X_test_fashion, y_test_fashion)


# %%
def create_fashion_conv_model():
    return nn.Sequential(
        # Image size: 28x28x1
        nn.Conv2d(1, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(32),
        nn.Dropout2d(0.6),
        # Input size: 28x28x32
        nn.Conv2d(32, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(32),
        nn.Dropout2d(0.6),
        # Input size: 28x28x32
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(64),
        nn.Dropout2d(0.6),
        # Input size: 28x28x64
        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(128),
        nn.Dropout2d(0.6),
        # Input size: 28x28x128
        nn.MaxPool2d(2),
        # Input size: 14x14x128
        nn.Flatten(1),
        nn.Linear(14 * 14 * 128, 1024),
        nn.ReLU(),
        nn.BatchNorm1d(1024),
        nn.Dropout(0.6),
        nn.Linear(1024, 10),
        nn.Softmax(dim=1),
    )


# %%
fashion_cnn_2 = NeuralNetClassifier(
    create_fashion_conv_model,
    max_epochs=40,
    lr=0.0025,
    optimizer=torch.optim.Adam,
    device=DEVICE,
    callbacks=[
        (
            "lr_scheduler",
            # LRScheduler(policy="StepLR", step_size=12, gamma=0.5)
            LRScheduler(policy="ExponentialLR", gamma=0.95)
        ),
    ],
)

# %%
fashion_cnn_2.fit(X_train_fashion, y_train_fashion)

# %%
evaluate_model(fashion_cnn_2, X_test_fashion, y_test_fashion)
