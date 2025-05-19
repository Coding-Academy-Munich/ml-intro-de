# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>Motivation für NumPy</b>
# </div>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# <br/>

# %% [markdown]
#
# # Vektoren und Matrizen als Python Listen

# %%
vector1 = [3, 2, 4]
vector2 = [8, 9, 7]

# %% [markdown]
#
# ## Berechnungen mit Python Listen
#
# Wir können für Berechnungen keine mathematischen Operatoren verwenden, da
# diese entweder nicht definiert sind, oder die falsche Bedeutung haben:

# %%
vector1 + vector2


# %%
# vector1 * vector2

# %%
# vector1 @ vector2

# %% [markdown]
#
# Statt Operatoren zu verwenden, können wir Funktionen definieren:

# %%
def vector_add(v1, v2):
    assert len(v1) == len(v2)
    result = [0] * len(v1)
    for i in range(len(v1)):
        result[i] = v1[i] + v2[i]
    return result


# %%
vector_add(vector1, vector2)

# %% [markdown]
#
# Matrizen können als verschachtelte Listen repräsentiert werden:
#
# $$
# M_1 = \begin{pmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \\ 7 & 8 & 9 \end{pmatrix}
# \qquad
# M_2 = \begin{pmatrix} 10 & 11 & 12 \\ 13 & 14 & 15 \\ 16 & 17 & 18 \end{pmatrix}
# $$


# %%
matrix1 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
matrix2 = [[10, 11, 12], [13, 14, 15], [16, 17, 18]]


# %%
def matrix_add(m1, m2):
    # Can only add matrices with the same number of rows
    assert len(m1) == len(m2)
    result = []
    for row1, row2 in zip(m1, m2):
        result.append(vector_add(row1, row2))
    return result


# %%
matrix_add(matrix1, matrix2)

# %% [markdown]
#
# ## Zugriff auf Elemente

# %%
vector1 = [3, 2, 4]

# %%
vector1[1]

# %% [markdown]
#
# Bei Matrizen kann mit mehreren Indexing-Operationen auf Elemente zugegriffen werden:

# %%
matrix1 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

# %%
matrix1[1]

# %%
matrix1[1][0]

# %% [markdown]
#
# ## Performance

# %%
from random import random


# %%
def make_random_vector(num_elts: int):
    return [round(random() * 100, 1) for _ in range(num_elts)]


# %%
make_random_vector(10)


# %%
def make_random_matrix(num_rows: int, num_cols: int):
    result = []
    for _ in range(num_rows):
        result.append(make_random_vector(num_cols))
    return result


# %%
make_random_matrix(5, 3)

# %%
m1 = make_random_matrix(20, 5)
m2 = make_random_matrix(20, 5)

# %%
matrix_add(m1, m2)[1]

# %%
from timeit import Timer


# %%
def compute_runtime_in_ms(code: str, num_iterations: int, repeat=1):
    timer = Timer(code, globals=globals())
    print("Starting timer run...", end="")
    time_in_ms = (
        min(timer.repeat(number=num_iterations, repeat=repeat)) / num_iterations * 1000
    )
    print("done.")
    print(f"Time per iteration: {time_in_ms:.2f}ms ({num_iterations} iterations).")
    return time_in_ms


# %%
m1 = make_random_matrix(10_000, 10)
m2 = make_random_matrix(10_000, 10)

# %%
time_python = compute_runtime_in_ms("matrix_add(m1, m2)", 100, 4)

# %%
import numpy as np

# %%
a1 = np.array(m1)
a2 = np.array(m2)

# %%
a1

# %%
a1 + a2

# %%
np.all(matrix_add(m1, m2) == a1 + a2)

# %%
time_numpy = compute_runtime_in_ms("a1 + a2", 10_000, 4)

# %%
print(f"Ratio python/numpy: {time_python/time_numpy:.1f}")

# %%
