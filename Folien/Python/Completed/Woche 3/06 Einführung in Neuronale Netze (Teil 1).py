# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>Einführung in Neuronale Netze (Teil 1)</b>
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
import matplotlib.pyplot as plt
from llm_utils import plot_neuron_2d


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
