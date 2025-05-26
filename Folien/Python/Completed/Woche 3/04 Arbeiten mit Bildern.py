# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>Arbeiten mit Bildern</b>
# </div>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# <br/>

# %%
from itertools import permutations

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# %%
colors = Image.open("imgdata/colors.png")

# %%
colors

# %%
type(colors)

# %%
plt.imshow(colors)

# %%
color_array = np.array(colors)

# %%
type(color_array)

# %%
color_array.shape

# %%
color_array

# %%
color_array[:1]

# %%
plt.imshow(color_array[:1])

# %%
plt.imshow(color_array[:, :1])

# %%
plt.imshow(color_array[0])

# %%
color_array[0].shape

# %%
plt.imshow(color_array[0].reshape(1, 4, 3))

# %%
plt.imshow(np.expand_dims(color_array[0], axis=0))

# %%
layers = np.split(color_array, 3, axis=2)
len(layers)

# %%
layers[0].shape

# %%
plt.imshow(layers[0], cmap="binary")

# %%
list(permutations([1, 2, 3]))

# %%
fig, axes = plt.subplots(2, 3)
lin_axes = axes.reshape(-1)
for i, p in enumerate(permutations(layers)):
    lin_axes[i].imshow(np.concatenate(p, axis=2))


# %% [markdown]
#
# ## Workshop: Transformieren eines Katzenbilds
#
# Die Datei `imgdata/tabby-cat-64.png` enthält ein Katzenbild.
#
# - Schreiben Sie eine Funktion `permute_colors(image)`, die alle möglichen
#   Permutationen der Farbkanäle des Bildes generiert und diese anzeigt.

# %%
tabby_cat = Image.open("imgdata/tabby-cat-64.png")

# %%
def permute_colors(image):
    layers = np.split(np.array(image), 3, axis=2)
    fig, axes = plt.subplots(2, 3)
    lin_axes = axes.reshape(-1)
    for i, p in enumerate(permutations(layers)):
        lin_axes[i].imshow(np.concatenate(p, axis=2))
    plt.show()


# %%
permute_colors(tabby_cat)

# %% [markdown]
#
# - Skalieren Sie das Bild auf eine Größe von 32x32 Pixeln.
# - Nehmen Sid dabei den mittelwert der Pixelwerte jedes 2x2-Fensters um den
#   Wert des neuen Pixels zu berechnen.
# - Verwenden Sie dabei numpy-Funktionen, um die Berechnungen durchzuführen.

# %%
tabby_cat_arr = np.array(tabby_cat)
new_shape = (32, 2, 32, 2, 3)
reshaped_tabby_cat = tabby_cat_arr.reshape(new_shape)
reshaped_tabby_cat[::, 0, ::, 0, ::].shape

# %%
fig, axes = plt.subplots(2, 2, figsize=(8, 8))
for x in range(2):
    for y in range(2):
        axes[x, y].imshow(
            reshaped_tabby_cat[:, x, :, y, :]
        )
        axes[x, y].axis("off")

# %%
def scale_image_to_half_size(image):
    arr = np.array(image)
    # Calculate the mean of each 2x2 block
    new_shape = (arr.shape[0] // 2, arr.shape[1] // 2, arr.shape[2])
    scaled_arr = arr.reshape(new_shape[0], 2, new_shape[1], 2, new_shape[2]).mean(axis=(1, 3))
    return Image.fromarray(scaled_arr.astype(np.uint8))


# %%
small_tabby_cat = scale_image_to_half_size(tabby_cat)

# %%
# Show the original and scaled images side by side
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(tabby_cat)
axes[0].set_title("Original Image")
axes[1].imshow(small_tabby_cat)
axes[1].set_title("Scaled Image (32x32)")

# %%
