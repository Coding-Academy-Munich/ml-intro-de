# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>Download der MNIST Daten</b>
# </div>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# <br/>

# %% [markdown]
#
# # Download der MNIST Daten
#
# Da der Download der MNIST Daten von OpenML relativ lange dauert, speichern wir
# eine lokale Version der Daten.
#
# Die `EnvConfig`-Klasse hilft uns dabei, die Verzeichnisse, in denen wir Daten
# speichern, zwischen verschiedenen Notebooks konsistent zu halten.

# %%
from envconfig import EnvConfig

# %%
config = EnvConfig()

# %%
print(f"Scikit-learn Cache Path:   {config.sklearn_cache_dir_path}")
print(f"Path to MNIST pickle file: {config.mnist_pkl_path}")

# %% [markdown]
#
# ## Herunterladen der Daten
#
# [Scikit-learn](https://scikit-learn.org/stable/index.html) ist eine Open-Source
# Bibliothek, die viele traditionelle Algorithmen für maschinelles Lernen implementiert.
#
# Außerdem stellt sie viele Hilfsmittel zur Verarbeitung von Daten, Modell-Auswahl,
# Modell-Evaluierung, usw. bereit.
#
# In diesem Notebook verwenden wir einen der Beispiel-Datensätze, die scikit-learn
# anbietet. Der MNIST-Datensatz ist das typische "Hello, world!" Beispiel für
# Algorithmen zum maschinellen Lernen.

# %%
from sklearn.datasets import fetch_openml

# %%
print("Loading MNIST data...", flush=True, end="")
mnist = fetch_openml(
    "mnist_784",
    version=1,
    data_home=config.sklearn_cache_dir_path.as_posix(),
    cache=True,
    parser="auto",
)
print("done.", flush=True)

# %% [markdown]
#
# ### Schreiben der Daten

# %%
config.mnist_pkl_path.parent.mkdir(exist_ok=True, parents=True)

# %%
import pickle

# %%
with open(config.mnist_pkl_path, "wb") as file:
    print(f"Writing MNIST data to {config.mnist_pkl_path}.")
    pickle.dump(mnist, file)

# %%
assert config.mnist_pkl_path.exists()

# %%
print("Downloaded and saved MNIST data.")

# %% [markdown]
#
# ## Mini-Workshop: Download der Fashion-MNIST Daten
#
# [Fashion-MNIST](https://www.openml.org/search?type=data&sort=runs&id=40996)
# ist ein Datensatz, der im gleichen Format wie MNIST ist, aber Bilder
# von verschiedenen Kleidungsstücken enthält.
#
# - Laden Sie den Fashion-MNIST Datensatz von der OpenML Website herunter:<br>
#   `fetch_openml(data_id=40996)` oder<br>
#   `fetch_openml("Fashion-MNIST", version=1)`
# - Schreiben Sie die Daten in eine Pickle-Datei
#
# *Hinweis:* Die `EnvConfig`-Klasse enthält dafür ein Attribut `fashion_mnist_pkl_path`.


# %%
from envconfig import EnvConfig

# %%
import pickle

from sklearn.datasets import fetch_openml

# %%
config = EnvConfig()

# %%
print(f"Path to Fashion-MNIST pickle file: {config.fashion_mnist_pkl_path}")

# %%
fashion_mnist = fetch_openml(
    data_id=40996,
    data_home=config.sklearn_cache_dir_path.as_posix(),
    cache=True,
    parser="auto",
)

# %%
print("Loading Fashion-MNIST data (by name)...", flush=True, end="")
fashion_mnist_by_name = fetch_openml(
    "Fashion-MNIST",
    version=1,
    data_home=config.sklearn_cache_dir_path.as_posix(),
    cache=True,
)
print("done.", flush=True)

# %%
config.fashion_mnist_pkl_path.parent.mkdir(exist_ok=True, parents=True)

# %%
with open(config.fashion_mnist_pkl_path, "wb") as file:
    print(f"Writing Fashion-MNIST data to {config.fashion_mnist_pkl_path}.")
    pickle.dump(fashion_mnist, file)

# %%
assert config.fashion_mnist_pkl_path.exists()

# %%
print("Done.")

# %%
