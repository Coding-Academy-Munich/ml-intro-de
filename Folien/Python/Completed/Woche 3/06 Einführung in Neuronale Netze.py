# %%
# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>Einführung in Neuronale Netze</b>
# </div>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# <br/>

# %% [markdown]
# <img src="img/Figure-01-021.png" style="float: center; width: 30%; margin-left: auto; margin-right: auto;"/>

# %% [markdown]
# ## Neuronen
#
# <img src="img/Figure-10-001.png" style="width: 80%; margin-left: auto; margin-right: auto;"/>

# %% [markdown]
# ## Künstliche Neuronen
#
# <img src="img/Figure-10-006.png" style="width: 60%; margin-left: auto; margin-right: auto;"/>

# %% [markdown]
# ## Künstliche Neuronen
#
# <img src="img/Figure-10-004.png" style="width: 30%; margin-left: auto; margin-right: auto;"/>

# %% [markdown]
# ## Lineare Modelle

# %%
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skorch import NeuralNetClassifier
from skorch.callbacks import EpochScoring


# %%
def linear_model(x):
    return 2 * x + 1

# %%
linear_model(3)

# %%
linear_model(torch.tensor([1.0, 2.0, 3.0]))


# %%
torch.linspace(-2, 2, 5)

# %%
torch.linspace(-2, 1, 7)

# %%
x = torch.linspace(-6, 6, 100)

# %%
x[0], x[-1]

# %%
x[:1]

# %%
plt.plot(x, linear_model(x));

# %%
lin = nn.Linear(1, 1)

# %%
lin

# %%
lin.bias

# %%
lin.weight

# %%
lin(x[:1])

# %%
lin(x.view(100, 1))[:4]

# %%
plt.figure()
plt.plot(x, lin(x.view(100, 1)).detach());

# %%
nn.Linear(1, 1).weight

# %% [markdown]
# ## Aktivierungsfunktionen

# %%
plt.figure(figsize=(10, 12))
plt.subplot(3, 2, 1)
plt.plot(x, nn.Sigmoid()(x))
plt.subplot(3, 2, 2)
plt.plot(x, nn.Tanh()(x))
plt.subplot(3, 2, 3)
plt.plot(x, nn.ReLU()(x))
plt.subplot(3, 2, 4)
plt.plot(x, - nn.ReLU()(x + 2))
plt.subplot(3, 2, 5)
plt.plot(x, nn.ReLU()(x) - nn.ReLU()(x + 2))
plt.subplot(3, 2, 6)
plt.plot(x, nn.Tanh()(x) - 1.5 * nn.Tanh()(x - 2));
plt.show()


# %%
def neuron1(x):
    return nn.Tanh()(nn.Linear(1, 1)(x))


# %%
plt.figure()
plt.plot(x, neuron1(x.view(100, 1)).detach());


# %%
def neuron2(xy):
    return nn.Tanh()(nn.Linear(2, 1)(xy))


# %%
def plot_neuron_2d(neuron):
    # Create a grid of input points
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)

    # Prepare input for the neuron
    xy = torch.tensor(np.column_stack((X.ravel(), Y.ravel())), dtype=torch.float32)

    # Pass the input through the neuron
    with torch.no_grad():
        Z = neuron(xy).numpy().reshape(X.shape)

    # Create the 3D plot
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the surface
    surf = ax.plot_surface(X, Y, Z, cmap='viridis')

    # Add a color bar
    # fig.colorbar(surf, shrink=0.5, aspect=5)

    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Activation')
    ax.set_title('2D Neuron Activation')

    # Enable rotation
    ax.mouse_init()

    # Adjust the view angle for better initial visualization
    ax.view_init(elev=20, azim=45)

    # Tight layout to ensure everything fits
    plt.tight_layout()

    # Show the plot
    plt.show()



# %%
plot_neuron_2d(neuron2)


# %%
# Define the neuron
class Neuron2D(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)
        self.activation = nn.Tanh()

    def forward(self, xy):
        return self.activation(self.linear(xy))


# %%
plot_neuron_2d(Neuron2D())

# %% [markdown]
# ## Neuronale Netze
#
# <img src="img/Figure-18-032.png" style="width: 70%; margin-left: auto; margin-right: auto;"/>

# %%
seq_model = nn.Sequential(
    nn.Linear(2, 4),
    nn.ReLU(),
    nn.Linear(4, 3),
    nn.ReLU(),
    nn.Linear(3, 2)
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

# %%
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils.data import DataLoader
from skorch.dataset import Dataset

# %%
input_size = 28 * 28
num_classes = 10
num_epochs = 5
batch_size = 100
learning_rate = 0.005

# %%
mnist_transforms = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor()
])

# %%
train_dataset = torchvision.datasets.MNIST(root='./data',
                                           train=True,
                                           transform=mnist_transforms,
                                           download=True)
test_dataset = torchvision.datasets.MNIST(root='./data',
                                          train=False,
                                          transform=mnist_transforms,
                                          download=True)

# %%
it = iter(train_dataset)
next(it)[0].shape, next(it)[1]

# %%
next(it)[0].shape, next(it)[1]

# %%
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


# %%
class MnistModule(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, X, **kwargs):
        return self.model(X.reshape(-1, input_size))


# %%
def create_skorch_model(hidden_size, epochs):
    model = NeuralNetClassifier(
        MnistModule,
        module__input_size=input_size,
        module__hidden_size=hidden_size,
        module__num_classes=num_classes,
        max_epochs=epochs,
        lr=learning_rate,
        optimizer=torch.optim.Adam,
        criterion=nn.CrossEntropyLoss,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        iterator_train__shuffle=True,
        train_split=None,  # We'll use our own validation set
        callbacks=[
            EpochScoring(scoring='accuracy', lower_is_better=False, name='val_acc', on_train=False),
        ],
        verbose=1,
    )
    return model


# %%
model = create_skorch_model(32, 5)


# %%
class MnistDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data.dataset)

    def __getitem__(self, i):
        x, y = self.data.dataset[i]
        return x.reshape(-1, input_size), y


# %%
train_dataset = MnistDataset(train_loader)

# %%
test_dataset = MnistDataset(test_loader)

# %%
model.fit(train_dataset, y=None, X_valid=test_dataset, y_valid=None)


# %%
def fit_model(hidden_size, num_epochs=num_epochs):
    model = create_skorch_model(hidden_size, num_epochs)

    train_dataset = MnistDataset(train_loader)
    test_dataset = MnistDataset(test_loader)

    model.fit(train_dataset, y=None, X_valid=test_dataset, y_valid=None)

    return model


# %%
def evaluate_model(model, test_dataset):
    y_pred = model.predict(test_dataset)
    y_true = torch.cat([y for _, y in test_loader], dim=0).numpy()

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall: {recall:.4f}")
    print(f"Test F1 Score: {f1:.4f}")

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

    return accuracy, precision, recall, f1


# %%
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns


# %%
def fit_and_evaluate_model(hidden_size, num_epochs):
    model = fit_model(hidden_size, num_epochs)
    losses = model.history[:, 'train_loss']
    test_dataset = MnistDataset(test_loader)

    plt.figure(figsize=(6, 3))
    plt.plot(range(len(losses)), losses)
    plt.title(f'Hidden Size: {hidden_size}, Epochs: {num_epochs}')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.show()

    evaluate_model(model, test_dataset)


# %%
fit_and_evaluate_model(32, 5)

# %%
fit_and_evaluate_model(128, 8)

# %%
fit_and_evaluate_model(512, 10)


# %% [markdown]
# ## Modelle
#
# <img src="img/Figure-11-001.png" style="width: 100%;"/>

# %% [markdown]
# ## Für Neuronale Netze:
#
# Was repräsentiert werden kann hängt ab von
#
# - Anzahl der Layers
# - Anzahl der Neutronen per Layer
# - Komplexität der Verbindungen zwischen Neutronen

# %% [markdown]
# ### Was kann man (theoretisch) lernen?
#
# Schwierig aber irrelevant

# %% [markdown]
# ### Was kann man praktisch lernen?
#
# Sehr viel, wenn man genug Zeit und Daten hat

# %% [markdown]
# ### Was kann man effizient lernen?
#
# Sehr viel, wenn man sich geschickt anstellt
# (und ein Problem hat, an dem viele andere Leute arbeiten)

# %% [markdown]
# # Bias/Variance Tradeoff
#
# - Modelle mit geringer Expressivität (representational power)
#   - Können schnell trainiert werden
#   - Arbeiten mit wenig Trainingsdaten
#   - Sind robust gegenüber Fehlern in den Trainingsdaten
#
# - Wir sind nicht an einer möglichst exakten Wiedergabe unserer Daten interessiert
#
# - Entscheidend ist wie gut unser Modell auf unbekannte Daten generalisiert

# %% [markdown]
# <img src="img/Figure-09-002.png" style="width: 60%; margin-left: auto; margin-right: auto;"/>

# %% [markdown]
# <img src="img/Figure-09-004.png" style="width: 60%; margin-left: auto; margin-right: auto;"/>

# %% [markdown]
# <img src="img/Figure-09-003.png" style="width: 60%; margin-left: auto; margin-right: auto;"/>

# %% [markdown]
# <img src="img/Figure-09-005.png" style="width: 60%; margin-left: auto; margin-right: auto;"/>

# %% [markdown]
#
# ### Generalisierung und Rauschen
# <img src="img/Figure-09-008.png" style="width: 40%; margin-left: auto; margin-right: auto;"/>

# %% [markdown]
# <img src="img/Figure-09-009.png" style="width: 80%; margin-left: auto; margin-right: auto;"/>

# %% [markdown]
# <img src="img/Figure-09-010.png" style="width: 40%; margin-left: auto; margin-right: auto;"/>

# %% [markdown]
# ## Komplexität der Entscheidungsgrenze
#
# <img src="img/Figure-09-006.png" style="width: 100%; margin-left: auto; margin-right: auto;"/>

# %% [markdown]
# <img src="img/Figure-09-001.png" style="width: 50%; margin-left: auto; margin-right: auto;"/>

# %% [markdown]
# ## Datenverteilung und Qualität
#

# %% [markdown]
# ### Erinnerung: die Trainings-Schleife
#
# <img src="img/Figure-08-001.png" style="width: 20%; margin-left: auto; margin-right: auto;"/>

# %% [markdown]
# <img src="img/Figure-08-001.png" style="width: 60%; margin-left: auto; margin-right: auto;"/>

# %% [markdown]
# ## Was lernt ein Klassifizierer?
#
# <img src="img/Figure-08-002.png" style="width: 60%; margin-left: auto; margin-right: auto;"/>

# %% [markdown]
# <img src="img/Figure-08-003.png" style="width: 100%; margin-left: auto; margin-right: auto;"/>

# %% [markdown]
# <img src="img/Figure-08-004.png" style="width: 70%; margin-left: auto; margin-right: auto;"/>

# %% [markdown]
# # Wie gut sind wir?
#
# Wie wissen wir, wie gut unser Modell wirklich ist?

# %% [markdown]
# ## Was kann schief gehen?
#
# <img src="img/Figure-03-015.png" style="width: 100%; margin-left: auto; margin-right: auto; 0"/>

# %% [markdown]
# ## Was kann schief gehen?
#
# <img src="img/Figure-03-017.png" style="width: 100%; margin-left: auto; margin-right: auto; 0"/>

# %% [markdown]
# ## Was kann schief gehen?
#
# <img src="img/Figure-03-018.png" style="width: 80%; margin-left: auto; margin-right: auto; 0"/>

# %% [markdown]
# ## Accuracy: Wie viel haben wir richtig gemacht?
#
#
# <img src="img/Figure-03-023.png" style="width: 60%; margin-left: auto; margin-right: auto; 0"/>

# %% [markdown]
# ## Precision: Wie gut sind unsere positiven Elemente?
#
#
# <img src="img/Figure-03-024.png" style="width: 60%; margin-left: auto; margin-right: auto; 0"/>

# %% [markdown]
# ## Recall: Wie viele positive Elemente haben wir übersehen?
#
#
# <img src="img/Figure-03-026.png" style="width: 60%; margin-left: auto; margin-right: auto; 0"/>

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
def create_conv_model():
    model = nn.Sequential(
        nn.Conv2d(1, 32, 3, 1),
        nn.ReLU(),
        nn.Conv2d(32, 64, 3, 1),
        nn.MaxPool2d(2),
        nn.Dropout2d(0.25),
        nn.Flatten(1),
        nn.Linear(9216, 128),
        nn.ReLU(),
        nn.Dropout2d(0.5),
        nn.Linear(128, 10)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    return model, optimizer

# %% [markdown]
# ## Data Engine (Tesla)
#
# <img src="img/data-engine.jpeg" style="width: 100%; margin-left: auto; margin-right: auto;"/>

# %% [markdown]
# ## Probleme: Abhängigkeiten
#
# Relevante Informationen sind nicht immer nahe in den Daten:
#
# "Er hatte mit dem Mann, der ihm den Schlüssel, der zum Schloss, das ihn von großem Reichtum trennte, gehörte, gab, noch nicht gesprochen.

# %% [markdown]
# # Memory / State
# <img src="img/Figure-22-012.png" style="width: 20%; margin-left: auto; margin-right: auto;"/>

# %% [markdown]
# Funktioniert gut aber mit gewissen Schwächen.
#
# Man muss wissen, welche Information für das aktuell betrachtete Element relevant ist:

# %% [markdown]
# - The cat didn't cross the street because *it* was too wide.

# %% [markdown]
# <img src="img/garfield.jpg" style="float: right;width: 60%;"/>

# %% [markdown]
# <img src="img/garfield-yawn.png" style="float: right;width: 60%;"/>
#
# - The cat didn't cross the street because *it* was too tired.

# %% [markdown]
# - The cat didn't cross the street because *it* was too wet.
#
# <img src="img/garfield-rain4.jpg" style="float: right;width: 60%;"/>

# %% [markdown]
# - The cat didn't cross the street because *it* was raining.
#
# <img src="img/garfield-rain2.gif" style="float: right;width: 60%;"/>

# %% [markdown]
# # The Bitter Lesson (Rich Sutton)
#
# [T]he only thing that matters in the long run is the leveraging of computation.
#
# Corollary: And data. Lots of data.

# %%
