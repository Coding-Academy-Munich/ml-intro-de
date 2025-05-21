# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>Einführung in Neuronale Netze (Teil 2)</b>
# </div>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# <br/>

# %%
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import matplotlib.pyplot as plt
from skorch import NeuralNetClassifier
from skorch.callbacks import EpochScoring
from skorch.dataset import Dataset
from llm_utils import plot_neuron_2d, evaluate_model

# %% [markdown]
# ## Neuronale Netze
#
# <img src="img/Figure-18-032.png" style="width: 70%; margin-left: auto; margin-right: auto;"/>

# %%
seq_model = nn.Sequential(
    nn.Linear(2, 4), nn.ReLU(), nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2)
)

# %%
seq_model(torch.tensor([1.0, 2.0]))


# %%
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(2, 12)
        self.linear2 = nn.Linear(12, 8)
        self.linear3 = nn.Linear(8, 4)
        self.linear4 = nn.Linear(4, 1)
        self.activation_fun = nn.Tanh

    def forward(self, x):
        y = self.linear1(x)
        y = self.activation_fun()(y)
        y = self.linear2(y)
        y = self.activation_fun()(y)
        y = self.linear3(y)
        y = self.activation_fun()(y)
        y = self.linear4(y)
        y = self.activation_fun()(y)
        return y


# %%
plot_neuron_2d(SimpleNet())

# %% [markdown]
# ## Erinnerung: Training
#
# <br/>
# <img src="img/Figure-01-008.png" style="width: 100%;"/>

# %% [markdown]
# ## Training Neuraler Netze
#
# <img src="img/Figure-18-033.png" style="width: 100%; margin-left: auto; margin-right: auto;"/>

# %% [markdown]
#
# ## Training Neuraler Netze
#
# <img src="img/Figure-18-034.png" style="width: 40%; margin-left: auto; margin-right: auto;"/>

# %% [markdown]
# ## Training Neuraler Netze
#
# <img src="img/Figure-18-035.png" style="width: 100%; margin-left: auto; margin-right: auto; 0"/>

# %% [markdown]
# ## Training Neuraler Netze
#
# <img src="img/Figure-18-036.png" style="width: 50%; margin-left: auto; margin-right: auto;"/>

# %% [markdown]
# ## Training Neuraler Netze
#
# <img src="img/Figure-18-037.png" style="width: 100%; margin-left: auto; margin-right: auto; 0"/>

# %% [markdown]
# ## Wie updaten wir die Parameter?
#
# <img src="img/Figure-05-012.png" style="width: 35%; margin-left: auto; margin-right: auto; 0"/>

# %% [markdown]
# ## Wie updaten wir die Parameter?
#
# <img src="img/Figure-05-013.png" style="width: 60%; margin-left: auto; margin-right: auto; 0"/>

# %% [markdown]
# ## Wie updaten wir die Parameter?
#
# <img src="img/Figure-05-001.png" style="float: left; width: 45%; margin-left: auto; margin-right: auto; 0"/>
# <img src="img/Figure-05-005.png" style="float: right; width: 45%; margin-left: auto; margin-right: auto; 0"/>

# %% [markdown]
# # MNIST
#
# Im Folgenden wollen wir ein einfaches neuronales Netz auf dem MNIST-Datensatz
# trainieren.

# %% [markdown]
#
# ### Modell- und Trainings-Parameter
#
# Wir definieren einige Parameter für das Modell und das Training

# %%
INPUT_SIZE = 28 * 28
NUM_CLASSES = 10
NUM_EPOCHS = 5
BATCH_SIZE = 100
LEARNING_RATE = 0.005
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# %%
DEVICE

# %% [markdown]
#
# ### Vorbereiten des Datasets
#
# - Um den MNIST-Datensatz zu verwenden, müssen wir ihn zuerst herunterladen
# - Das machen wir mit `torchvision.datasets.MNIST`
# - Dabei geben wir auch Transformationen an, die auf die Bilder angewendet
#   werden
# - In diesem Fall verwenden wir `transforms.Resize` und `transforms.ToTensor`
#   - `transforms.Resize` ändert die Größe der Bilder auf 28x28 Pixel
#   - `transforms.ToTensor` wandelt die Bilder in Tensoren um
# - Wir könnten noch weitere Transformationen hinzufügen

# %%
mnist_transforms = transforms.Compose(
    [transforms.Resize((28, 28)), transforms.ToTensor()]
)


# %%
augmented_transforms = transforms.Compose(
    [transforms.Resize((28, 28)), transforms.ToTensor(), transforms.RandomRotation(25)]
)

# %%
train_dataset = torchvision.datasets.MNIST(
    root="./localdata", train=True, transform=mnist_transforms, download=True
)

# %%
augmented_train_dataset = torchvision.datasets.MNIST(
    root="./localdata", train=True, transform=augmented_transforms, download=True
)

# %%
test_dataset = torchvision.datasets.MNIST(
    root="./localdata", train=False, transform=mnist_transforms, download=True
)

# %%
type(train_dataset)

# %%
len(train_dataset)

# %%
type(train_dataset[0])

# %%
x, y = train_dataset[0]
x.shape, x.min(), x.max(), y

# %%
x[0, 12:18, 12:18]

# %% [markdown]
#
# Erzeugen eines Loaders aus dem Dataset
# - Ein Torch `DataLoader` ist ein Iterator, der das Dataset in Batches aufteilt
# - Erzeugt Batches von `batch_size` Elementen
# - `shuffle=True` mischt die Daten
# - `shuffle=False` gibt die Daten in der Reihenfolge zurück, wie sie im Dataset sind

# %%
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True
)

# %%
augmented_train_loader = torch.utils.data.DataLoader(
    dataset=augmented_train_dataset, batch_size=BATCH_SIZE, shuffle=True
)

# %%
test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False
)


# %% [markdown]
#
# ### Definieren des Modells
#
# Wir definieren ein einfaches neuronales Netz mit einer einzigen versteckten
# Schicht
#
# - `input_size` ist die Anzahl der Eingabewerte
# - `hidden_size` ist die Anzahl der Neuronen in der versteckten Schicht
# - `num_classes` ist die Anzahl der Ausgabewerte

# %%
class MnistModule(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, X, **kwargs):
        return self.model(X.reshape(-1, INPUT_SIZE))


# %% [markdown]
#
# ### Definieren des Skorch-Modells
#
# Wir definieren ein Skorch-Modell, das die oben definierte Klasse verwendet
# - `NeuralNetClassifier` ist das Torch-Modell
# - Mit `module___...` können wir die Parameter des Modells setzen
# - `max_epochs` ist die Anzahl der Epochen
# - `lr` ist die Lernrate
# - `optimizer` ist der Optimierer
# - `criterion` ist die Verlustfunktion
# - `device` ist das Gerät, auf dem das Modell trainiert wird
# - `iterator_train__shuffle=True` mischt die Daten
# - `train_split=None` bedeutet, dass wir unser eigenes Validierungsset verwenden
# - `callbacks` sind die Rückrufe, die während des Trainings aufgerufen werden
# - `verbose=1` gibt den Fortschritt des Trainings aus

# %%
def create_skorch_model(hidden_size, epochs):
    model = NeuralNetClassifier(
        MnistModule,
        module__input_size=INPUT_SIZE,
        module__hidden_size=hidden_size,
        module__num_classes=NUM_CLASSES,
        max_epochs=epochs,
        lr=LEARNING_RATE,
        optimizer=torch.optim.Adam,
        criterion=nn.CrossEntropyLoss,
        device="cuda" if torch.cuda.is_available() else "cpu",
        iterator_train__shuffle=True,
        train_split=None,  # We'll use our own validation set
        callbacks=[
            EpochScoring(
                scoring="accuracy",
                lower_is_better=False,
                name="val_acc",
                on_train=False,
            ),
        ],
        verbose=1,
    )
    return model


# %% [markdown]
#
# ### Training des Modells
#
# - Wir verwenden die `fit`-Methode von Skorch, um das Modell zu trainieren
# - Wir geben dazu das Trainings- und Validierungsset an
# - Da unser Modell ein eindimensionales Bild erwartet, müssen wir die Bilder
#   umformen
# - Wir verwenden `MnistDataset`, um das Dataset in ein Format zu bringen, das
#   Skorch versteht

# %%
model = create_skorch_model(32, 3)


# %%
class MnistDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data.dataset)

    def __getitem__(self, i):
        x, y = self.data.dataset[i]
        return x.reshape(-1, INPUT_SIZE), y


# %%
train_dataset = MnistDataset(train_loader)

# %%
test_dataset = MnistDataset(test_loader)

# %%
model.fit(train_dataset, y=None, X_valid=test_dataset, y_valid=None)


# %%
def fit_model(hidden_size, num_epochs=NUM_EPOCHS, train_loader=train_loader):
    model = create_skorch_model(hidden_size, num_epochs)

    train_dataset = MnistDataset(train_loader)
    test_dataset = MnistDataset(test_loader)

    model.fit(train_dataset, y=None, X_valid=test_dataset, y_valid=None)

    return model


# %%
def fit_and_evaluate_model(
    hidden_size, num_epochs=NUM_EPOCHS, train_loader=train_loader
):
    model = fit_model(hidden_size, num_epochs)
    losses = model.history[:, "train_loss"]
    test_dataset = MnistDataset(test_loader)

    plt.figure(figsize=(6, 3))
    plt.plot(range(len(losses)), losses)
    plt.title(f"Hidden Size: {hidden_size}, Epochs: {num_epochs}")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.show()

    evaluate_model(model, test_dataset)


# %%
fit_and_evaluate_model(32, 5)

# %%
fit_and_evaluate_model(128, 8)

# %%
fit_and_evaluate_model(128, 8, augmented_train_loader)

# %%
fit_and_evaluate_model(512, 10)

# %%
fit_and_evaluate_model(512, 10, augmented_train_loader)

# %%
