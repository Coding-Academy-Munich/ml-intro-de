# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>Glückszahl 7 mit MNIST (Teil 1)</b>
# </div>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# <br/>

# %% [markdown]
#
# # Glückszahl 7 mit MNIST

# %%
from envconfig import EnvConfig

# %%
import pickle

import matplotlib.pyplot as plt
import numpy as np

# %%
config = EnvConfig()

# %%
with open(config.processed_mnist_pkl_path, "rb") as mnist_file:
    mnist_data = pickle.load(mnist_file)

# %%
x_train = mnist_data["x_train"]
x_test = mnist_data["x_test"]
y_train = mnist_data["y_train"]
y_test = mnist_data["y_test"]

# %%
lucky7_train = y_train == 7
lucky7_test = y_test == 7

# %%
lucky7_test[:3]

# %%
y_test[:3]

# %%
plt.imshow(x_test[0].reshape(28, 28), cmap="binary")
plt.show()


# %%
def show_as_images(x):
    fig, axes = plt.subplots(10, 10)
    for idx, ax in enumerate(np.array(axes).ravel()):
        ax.imshow(x[idx].reshape(28, 28), cmap="binary")
        ax.set_xticks([], [])
        ax.set_yticks([], [])
    plt.show()


# %%
show_as_images(x_test)


# %%
from sklearn.linear_model import SGDClassifier

# %%
sgd_clf = SGDClassifier(random_state=42)

# %% [markdown]
#
# ## Training des Modells
#
# - Mit der `fit()`-Methode trainieren wir das Modell.
# - Um die Trainingszeit zu verkürzen (mit schlechteren Ergebnissen), können wir
#   nur einen Teil der Trainingsdaten verwenden:

# %%
sgd_clf.fit(x_train, lucky7_train)

# %%
sgd_clf.predict([x_test[0]])

# %%
sgd_clf.predict(x_test[:3])

# %%
lucky7_predict = sgd_clf.predict(x_test)

# %%
correct_predictions = lucky7_predict == lucky7_test

# %%
correct_predictions[:10]

# %%
np.sum(correct_predictions)

# %% [markdown]
#
# ## Mini-Workshop: Linearer Klassifizierer für Fashion-MNIST
#
# - Trainieren Sie einen SGD Klassifizierer für den Fashion-MNIST Datensatz.
# - Bestimmen Sie, wie viele Sneaker (Kategorie 7) Sie damit richtig klassifizieren
#   können.


# %%
import pickle

import matplotlib.pyplot as plt  # noqa
import numpy as np  # noqa

# %%
from envconfig import EnvConfig

# %%
config = EnvConfig()

# %%
with open(config.processed_fashion_mnist_pkl_path, "rb") as file:
    fashion_mnist_data = pickle.load(file)

# %%
x_train = fashion_mnist_data["x_train"]
x_test = fashion_mnist_data["x_test"]
y_train = fashion_mnist_data["y_train"]
y_test = fashion_mnist_data["y_test"]

# %%
sneaker_train = y_train == 7
sneaker_test = y_test == 7

# %%
sneaker_test[:10], y_test[:10]

# %%
plt.imshow(x_test[0].reshape(28, 28), cmap="binary")
plt.show()

# %%
np.argmin(sneaker_test == False)  # noqa

# %%
plt.imshow(x_test[9].reshape(28, 28), cmap="binary")
plt.show()

# %%
from sklearn.linear_model import SGDClassifier  # noqa: E402

# %%
sgd_clf = SGDClassifier(random_state=42)

# %%
sgd_clf.fit(x_train, sneaker_train)

# %%
sgd_clf.predict([x_test[0]])

# %%
sgd_clf.predict([x_test[9]])

# %%
sgd_clf.predict(x_test[:10])

# %%
sneaker_predict = sgd_clf.predict(x_test)

# %%
correct_predictions = sneaker_predict == sneaker_test

# %%
correct_predictions[:10]

# %%
np.sum(correct_predictions)

