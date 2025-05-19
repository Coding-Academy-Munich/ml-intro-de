# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>Broadcasting</b>
# </div>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# <br/>

# %% [markdown]
#
# # Broadcasting
#
# Die meisten Operationen in Numpy können auch mit Skalaren ausgeführt werden:

# %%
import numpy as np

# %%
v1 = np.arange(8)
v1

# %%
v1 + 5

# %%
3 + v1

# %%
v1 * 2

# %%
v1**2

# %%
2**v1

# %%
v1 > 5

# %%
arr = np.arange(16).reshape((2, 2, 4))
arr

# %%
arr * arr

# %%
3 * arr

# %%
vec1 = np.arange(3)
vec1

# %%
vec1.shape

# %%
# arr * vec1

# %%
arr

# %%
arr.shape

# %%
vec2 = np.arange(4)
vec2

# %%
vec2.shape

# %%
res = arr * vec2

# %%
res.shape


# %% [markdown]
#
# ### Regeln für Broadcasting:
#
# Wenn eine Operation auf `a` und `b` durchgeführt wird:
#
# - Die Achsen (Shapes) von `a` und `b` werden von rechts nach links verglichen.
#
# - Wenn `a` und `b` die gleiche Länge für eine Achse haben, sind sie kompatibel
#
# - Wenn entweder `a` oder `b` die Länge 1 für eine Achse hat, wird es konzeptionell
#   entlang dieser Achse so oft wiederholt wie nötig.
#
# - Wenn `a` und `b` unterschiedliche Längen entlang einer Achse haben und
#   keines von beiden die Länge 1 hat, sind sie inkompatibel.
#
# - Das Array mit dem niedrigeren Rang wird so behandelt, als hätte es Rang
#   1 für die fehlenden Achsen, die fehlenden Achsen werden links angehängt

# %%
def ones(shape):
    return np.ones(shape, dtype=np.int32)


# %%
def tensor(shape):
    from functools import reduce
    from operator import mul

    size = reduce(mul, shape, 1)
    return np.arange(1, size + 1).reshape(*shape)


# %%
tensor((2, 3))

# %%
tensor((1, 3))

# %%
tensor((2, 1))

# %%
ones((1, 3)) + tensor((2, 1))

# %%
np.concatenate([tensor((2, 1))] * 3, axis=1)

# %%
ones((1, 3))

# %%
np.concatenate([ones((1, 3))] * 2, axis=0)

# %%
ones((1, 3)) + tensor((2, 1))

# %%
tensor((1, 3)) + ones((2, 1))

# %%
tensor((1, 3)) + tensor((2, 1))

# %%
tensor((2, 3, 4))

# %%
tensor((2, 3, 1))

# %%
tensor((2, 1, 4))

# %%
ones((2, 3, 1)) + tensor((2, 1, 4))

# %%
ones((2, 3, 1))

# %%
np.concatenate([ones((2, 3, 1))] * 4, axis=2)

# %%
tensor((2, 1, 4))

# %%
np.concatenate([tensor((2, 1, 4))] * 3, axis=1)

# %%
ones((2, 3, 1)) + tensor((2, 1, 4))

# %%
tensor((2, 3, 1)) + ones((2, 1, 4))

# %%
tensor((2, 3, 1)) + tensor((2, 1, 4))

# %%
tensor((3, 1)) + tensor((2, 1, 4))

# %%
tensor((3, 1))

# %%
tmp1 = np.concatenate([tensor((3, 1))] * 4, axis=1)
print("Shape:", tmp1.shape)
tmp1

# %%
tmp2 = tmp1.reshape((1, 3, 4))
print("Shape:", tmp2.shape)
tmp2

# %%
tmp3 = np.concatenate([tmp2] * 2, axis=0)
print("Shape:", tmp3.shape)
tmp3

# %%
tensor((2, 1, 4))

# %%
tmp4 = np.concatenate([tensor((2, 1, 4))] * 3, axis=1)
print("Shape:", tmp4.shape)
tmp4

# %%
tmp3

# %%
tmp4

# %%
tmp3 + tmp4

# %%
