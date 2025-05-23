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
from skorch import NeuralNetClassifier
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
seq_model

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
# <img src="img/Figure-18-034.png" style="width: 40%; margin-left: auto; margin-right: auto;"/>

# %% [markdown]
#
# <img src="img/Figure-18-035.png" style="width: 100%; margin-left: auto; margin-right: auto;"/>

# %% [markdown]
#
# <img src="img/Figure-18-036.png" style="width: 50%; margin-left: auto; margin-right: auto;"/>

# %% [markdown]
#
# <img src="img/Figure-18-037.png" style="width: 100%; margin-left: auto; margin-right: auto;"/>

# %% [markdown]
# ## Wie updaten wir die Parameter?
#
# <img src="img/Figure-05-012.png" style="width: 35%; margin-left: auto; margin-right: auto;"/>

# %% [markdown]
#
# <img src="img/Figure-05-013.png" style="width: 60%; margin-left: auto; margin-right: auto;"/>

# %% [markdown]
#
# <img src="img/Figure-05-001.png" style="float: left; width: 45%; margin-left: auto; margin-right: auto;"/>
# <img src="img/Figure-05-005.png" style="float: right; width: 45%; margin-left: auto; margin-right: auto;"/>

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
BATCH_SIZE = 100
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


# %%
import torch.nn.functional as F

# %%
LEARNING_RATE = 0.2
DROPOUT = 0.5
NUM_EPOCHS = 10


# %% [markdown]
#
# ### Definieren des Modells
#
# Wir definieren ein einfaches neuronales Netz mit einer einzigen versteckten
# Schicht
#
# - `hidden_size` ist die Anzahl der Neuronen in der versteckten Schicht
# - `dropout` ist die Dropout-Rate

# %%
class MnistModule(nn.Module):
    def __init__(self, hidden_size, dropout=DROPOUT):
        super().__init__()
        self.dropout = dropout
        self.hidden = nn.Linear(INPUT_SIZE, hidden_size)
        self.out = nn.Linear(hidden_size, NUM_CLASSES)

    def forward(self, X, **kwargs):
        y = self.hidden(X)
        y = F.relu(y)
        y = F.dropout(y, p=self.dropout, training=self.training)
        y = self.out(y)
        y = F.softmax(y, dim=-1)
        return y


# %% [markdown]
#
# ### Definieren des Skorch-Modells
#
# Wir definieren ein Skorch-Modell, das die oben definierte Klasse verwendet
# - `NeuralNetClassifier` ist das Torch-Modell
# - Mit `module__...` können wir die Parameter des Modells setzen
# - `max_epochs` ist die Anzahl der Epochen
# - `lr` ist die Lernrate
# - `device` ist das Gerät, auf dem das Modell trainiert wird

# %%
def create_skorch_model(hidden_size, epochs=NUM_EPOCHS, dropout=DROPOUT, seed=2025):
    torch.manual_seed(seed)
    model = NeuralNetClassifier(
        MnistModule,
        module__hidden_size=hidden_size,
        module__dropout=dropout,
        max_epochs=epochs,
        lr=LEARNING_RATE,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    return model


# %% [markdown]
#
# ### Training des Modells
#
# - Wir verwenden die `fit`-Methode von Skorch, um das Modell zu trainieren
# - Da Skorch an Scikit-Learn angelehnt ist, verwenden wir nicht die
#   PyTorch Data Loader, sondern übergeben das Dataset direkt
# - Mit `fit` wird das Modell trainiert und die Trainings- und
#   Validierungsverluste werden gespeichert

# %%
model = create_skorch_model(32, epochs=8, dropout=0.0)

# %%
X_train = train_dataset.data.reshape(-1, INPUT_SIZE) / 255.0
y_train = train_dataset.targets

# %%
type(X_train), type(y_train), X_train.shape, y_train.shape

# %%
X_test = test_dataset.data.reshape(-1, INPUT_SIZE) / 255.0
y_test = test_dataset.targets

# %%
model.fit(X_train, y_train)

# %%
model.predict(X_test[30:40])

# %%
y_test[30:40].numpy()

# %% [markdown]
#
# Berechnen der Indizes der falsch klassifizierten Samples

# %%
torch.where(torch.tensor(model.predict(X_test)) != y_test)[0][:10]


# %%
model = create_skorch_model(32, 20, dropout=0.0)
model.fit(X_train, y_train)

# %%
evaluate_model(model, X_test, y_test)

# %%
model = create_skorch_model(32, 20, dropout=0.5)
model.fit(X_train, y_train)

# %%
evaluate_model(model, X_test, y_test)

# %%
model = create_skorch_model(32, 20, dropout=0.1)
model.fit(X_train, y_train)

# %%
evaluate_model(model, X_test, y_test)

# %%
model = create_skorch_model(1024, 200, dropout=0.0)
model.fit(X_train[:1000], y_train[:1000])

# %%
evaluate_model(model, X_test, y_test)

# %%
model = create_skorch_model(128, 35, dropout=0.5)
model.fit(X_train, y_train)

# %%
evaluate_model(model, X_test, y_test)

# %%
model = create_skorch_model(512, 40, dropout=0.75)
model.fit(X_train, y_train)

# %%
evaluate_model(model, X_test, y_test)

# %%
