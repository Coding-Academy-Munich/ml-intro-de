# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>Evaluierung von Lucky 7</b>
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
from sklearn.linear_model import SGDClassifier  # noqa: E402

# %%
sgd_clf = SGDClassifier(random_state=42)

# %%
sgd_clf.fit(x_train, lucky7_train)

# %%
lucky7_predict = sgd_clf.predict(x_test)

# %%
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)

# %% [markdown]
#
# ## Accuracy
#
#  <img src="img/Figure-03-023.png" style="float: right; width: 60%;"/>

# %%
accuracy_score(lucky7_test, lucky7_predict)

# %%
balanced_accuracy_score(lucky7_test, lucky7_predict)

# %% [markdown]
#
# ## Precision
#
#  <img src="img/Figure-03-024.png" style="float: right; width: 60%;"/>

# %%
precision_score(lucky7_test, lucky7_predict)

# %% [markdown]
#
# ## Recall
#
#  <img src="img/Figure-03-026.png" style="float: right; width: 60%;"/>

# %%
recall_score(lucky7_test, lucky7_predict)

# %% [markdown]
#
# ## F1-Score
#
#  <img src="img/Figure-03-024.png" style="float: left; width: 40%;"/>
#  <img src="img/Figure-03-026.png" style="float: right; width: 40%;"/>

# %%
f1_score(lucky7_test, lucky7_predict)

# %% [markdown]
#
# ## Confusion Matrix
#
#  <img src="img/Figure-03-018.png" style="float: right; width: 70%;"/>

# %%
confusion_matrix(lucky7_test, lucky7_predict)

# %%
ConfusionMatrixDisplay.from_predictions(lucky7_test, lucky7_predict)
plt.show()

# %%
# Alternative colors:
ConfusionMatrixDisplay.from_predictions(lucky7_test, lucky7_predict, cmap="brg")
plt.show()

# %%
# Or:
ConfusionMatrixDisplay.from_predictions(lucky7_test, lucky7_predict, cmap="bwr_r")
plt.show()

# %%
ConfusionMatrixDisplay.from_predictions(lucky7_test, lucky7_predict, normalize="pred")
plt.show()

# %%
ConfusionMatrixDisplay.from_predictions(lucky7_test, lucky7_predict, normalize="true")
plt.show()

# %% [markdown]
#
# ## Classification Report

# %%
print(classification_report(lucky7_test, lucky7_predict))

# %% [markdown]
#
# ## Mini-Workshop: Evaluierung von Fashion-MNIST
#
# - Trainieren Sie einen `SGDClassifier` für Fashion-MNIST, der Sneaker (Kategorie 7)
#   erkennt.
# - Evaluieren Sie seine Performance.

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
from sklearn.linear_model import SGDClassifier

# %%
sgd_clf = SGDClassifier()

# %%
sgd_clf.fit(x_train, sneaker_train)

# %%
sgd_pred = sgd_clf.predict(x_test)

# %%
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

# %%
print(classification_report(sneaker_test, sgd_pred))

# %%
ConfusionMatrixDisplay.from_predictions(sneaker_test, sgd_pred, normalize="pred")
plt.show()

