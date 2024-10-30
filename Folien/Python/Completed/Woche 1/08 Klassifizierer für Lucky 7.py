# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>Klassifizierer für Lucky 7</b>
# </div>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# <br/>

# %%
import pickle

import matplotlib.pyplot as plt  # noqa
import numpy as np  # noqa

# %%
from envconfig import EnvConfig

# %%
config = EnvConfig()

# %%
with open(config.processed_mnist_pkl_path, "rb") as file:
    mnist_data = pickle.load(file)

# %%
x_train = mnist_data["x_train"]
x_test = mnist_data["x_test"]
y_train = mnist_data["y_train"]
y_test = mnist_data["y_test"]

# %%
lucky7_train = y_train == 7
lucky7_test = y_test == 7


# %%
import os
from sklearn.metrics import classification_report
from sklearn.linear_model import SGDClassifier

# %% [markdown]
#
# ## Drucken von Konfusionsmatrizen

# %%
from plot_confusion_matrices import *

# %% [markdown]
#
# ## Verringern der Trainingsdaten
#
# Um das Training zu beschleunigen, trainieren wir mit weniger Daten.

# %%
x_train_ori, y_train_ori = x_train, y_train
lucky7_train_ori = lucky7_train
x_train = x_train[:10_000]
y_train = y_train[:10_000]
lucky7_train = lucky7_train[:10_000]


# %% [markdown]
#
# ## Klassifikation mit Linearem Modell

# %%
sgd_clf = SGDClassifier()

# %%
sgd_clf.fit(x_train, lucky7_train)

# %%
sgd_pred = sgd_clf.predict(x_test)

# %%
print(classification_report(lucky7_test, sgd_pred))

# %%
print(classification_report(lucky7_train, sgd_clf.predict(x_train)))

# %%
plot_confusion_matrices(sgd_clf, x_train, x_test, lucky7_train, lucky7_test)

# %% [markdown]
#
# ## Klassifikation mit Entscheidungsbäumen

# %%
from sklearn.tree import DecisionTreeClassifier, plot_tree

# %%
dt_clf = DecisionTreeClassifier(max_depth=2)

# %%
dt_clf.fit(x_train, lucky7_train)

# %%
dt_pred = dt_clf.predict(x_test)

# %%
fig, ax = plt.subplots(figsize=(8, 4), dpi=200)
fig.tight_layout()
plot_tree(dt_clf, ax=ax)
plt.show()

# %%
print(classification_report(lucky7_test, dt_pred))

# %%
print(classification_report(lucky7_train, dt_clf.predict(x_train)))

# %%
plot_confusion_matrices(dt_clf, x_train, x_test, lucky7_train, lucky7_test)

# %%
plot_confusion_matrices(
    dt_clf, x_train, x_test, lucky7_train, lucky7_test, normalize="true"
)

# %%
dt_clf2 = DecisionTreeClassifier()

# %%
dt_clf2.fit(x_train, lucky7_train)

# %%
dt_pred2 = dt_clf2.predict(x_test)

# %%
print(classification_report(lucky7_test, dt_pred2))

# %%
print(classification_report(lucky7_train, dt_clf2.predict(x_train)))

# %%
plot_confusion_matrices(dt_clf2, x_train, x_test, lucky7_train, lucky7_test)

# %% [markdown]
#
# ## Klassifikation mit Random Forests

# %%
from sklearn.ensemble import RandomForestClassifier  # noqa: E402

# %%
rf_clf = RandomForestClassifier(random_state=42, n_jobs=os.cpu_count())

# %%
rf_clf.fit(x_train, lucky7_train)

# %%
rf_pred = rf_clf.predict(x_test)

# %%
print(classification_report(lucky7_test, rf_pred))

# %%
print(classification_report(lucky7_train, rf_clf.predict(x_train)))

# %%
plot_confusion_matrices(rf_clf, x_train, x_test, lucky7_train, lucky7_test)

# %% [markdown]
#
# ## Vorsicht bei unbalancierten Datensätzen!
#
# Die beiden Klassen in unserem Datensatz sind unterschiedlich groß.
# Das führt dazu, dass wir bei der Beurteilung unserer Qualitätsmaße vorsichtig sein
# müssen.

# %% [markdown]
#
# ## Mini-Workshop: Klassifizierer für Fashion-MNIST
#
# - Trainieren und evaluieren Sie folgende Klassifizierer für Sneaker
#   für den Fashion-MNIST Datensatz:
#   - `sklearn.linear_model.RidgeClassifier`
#   - `sklearn.tree.DecisionTreeClassifier`
#   - `sklearn.neighbors.KNeighborsClassifier`
#   - `sklearn.ensemble.RandomForestClassifier`
#   - `sklearn.ensemble.HistGradientBoostingClassifier`

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
from sklearn.metrics import classification_report, ConfusionMatrixDisplay


# %%
def train_and_evaluate_classifier(clf):
    banner = f"Training {type(clf).__name__}"
    print(f"\n{banner}", flush=True)
    print(f"{'=' * len(banner)}\n", flush=True)
    clf.fit(x_train, sneaker_train)
    pred = clf.predict(x_test)
    print(classification_report(sneaker_test, pred))
    print(classification_report(sneaker_train, clf.predict(x_train)))
    ConfusionMatrixDisplay.from_predictions(sneaker_test, pred)
    plt.show()


# %%
from sklearn.linear_model import RidgeClassifier

# %%
train_and_evaluate_classifier(RidgeClassifier())

# %%
from sklearn.tree import DecisionTreeClassifier

# %%
train_and_evaluate_classifier(DecisionTreeClassifier())

# %%
from sklearn.neighbors import KNeighborsClassifier

# %%
train_and_evaluate_classifier(KNeighborsClassifier(n_jobs=32))

# %%
from sklearn.ensemble import RandomForestClassifier

# %%
train_and_evaluate_classifier(RandomForestClassifier(n_jobs=32))

# %%
from sklearn.ensemble import HistGradientBoostingClassifier

# %%
train_and_evaluate_classifier(HistGradientBoostingClassifier())

