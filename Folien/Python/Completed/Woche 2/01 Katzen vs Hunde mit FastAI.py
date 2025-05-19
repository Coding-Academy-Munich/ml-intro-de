# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>Katzen vs. Hunde mit FastAI</b>
# </div>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# <br/>

# %%
from fastai.vision.all import *

# %%
path = untar_data(URLs.PETS) / "images"
path


# %%
def is_cat(x):
    return x[0].isupper()


# %%
def create_dls():
    return ImageDataLoaders.from_name_func(
        path,
        get_image_files(path),
        valid_pct=0.2,
        seed=42,
        label_func=is_cat,
        item_tfms=Resize(224),
    )


# %%
dls = create_dls()
dls.show_batch()

# %%
learn = vision_learner(dls, models.resnet34, metrics=error_rate)
learn.dls = create_dls()

# %%
learn.fine_tune(1)

# %%
learn.show_results()

# %%
