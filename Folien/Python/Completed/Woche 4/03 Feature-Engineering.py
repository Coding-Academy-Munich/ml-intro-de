# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>Feature-Engineering</b>
# </div>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# <br/>

# %% [markdown]
#
# # Feature-Engineering

# %%
import numpy as np
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
import evaluation_tools as et

# %%
sns.set_theme()

# %%
rng = np.random.default_rng(42)


# %%
def lin(x):
    return 0.85 * x - 1.5


# %%
def fun(x):
    return 0.1 * x**2 + 2 * np.sin(x) - 2


# %% [markdown]
#
# ### Trainieren eines linearen Modells

# %%
x = rng.uniform(size=(150, 1), low=0.0, high=10.0)
x_train, x_test = x[:100], x[100:]
x_plot = np.linspace(0, 10, 500)
x_train[:3]

# %%
y_lin_train = lin(x_train).reshape(-1)
y_lin_test = lin(x_test).reshape(-1)
y_fun_train = fun(x_train).reshape(-1)
y_fun_test = fun(x_test).reshape(-1)

# %% [markdown]
#
# Linearer Regressor für lineare Funktion

# %%
lr_lin = LinearRegression()
lr_lin.fit(x_train, y_lin_train)

# %%
lr_lin.coef_, lr_lin.intercept_

# %%
et.evaluate_regressor(LinearRegression, lin, x_train, x_test)

# %% [markdown]
#
# Linearer Regressor für nicht-lineare Funktion

# %%
et.evaluate_regressor(LinearRegression, fun, x_train, x_test)

# %% [markdown]
#
# ### Feature-Augmentation
#
# - Idee: Wir geben dem linearen Regressor mehr Features, die nicht-linear sind.
# - Idealerweise die Funktionen von $x$, die in der Funktion $f$ vorkommen.
# - $\mathit{lin}(x) = 0.85 \cdot x - 1.5$
# - $\mathit{fun}(x) = 0.1 \cdot x^2+ 2 \cdot \sin(x) - 2$

# %% [markdown]
#
# $\mathit{fun}(x) = 0.1 \cdot x^2+ 2 \cdot \sin(x) - 2$

# %%
x_train_aug = np.concatenate([x_train, x_train * x_train, np.sin(x_train)], axis=1)
x_train_aug[:3]

# %%
x_test_aug = np.concatenate([x_test, x_test * x_test, np.sin(x_test)], axis=1)

# %% [markdown]
#
# - Trainieren eines linearen Regressors mit augmentierten Features
# - Lineare Funktion (mit Rauschen) als Zielvariable


# %%
lr_aug_lin = LinearRegression()
lr_aug_lin.fit(x_train_aug, y_lin_train)

# %% [markdown]
# $\mathit{lin}(x) = 0.85 \cdot x - 1.5$

# %%
lr_aug_lin.coef_, lr_aug_lin.intercept_

# %%
lin_reg, lin_reg_rand, lin_reg_chaos = et.evaluate_regressor(
    LinearRegression, lin, x_train_aug, x_test_aug, use_x_train_for_plot=True
)

# %% [markdown]
# $\mathit{lin}(x) = 0.85 \cdot x - 1.5$

# %%
lin_reg.coef_, lin_reg.intercept_

# %%
lin_reg_rand.coef_, lin_reg_rand.intercept_

# %%
lin_reg_chaos.coef_, lin_reg_chaos.intercept_

# %% [markdown]
#
# - Trainieren eines linearen Regressors mit augmentierten Features
# - Nicht-lineare Funktion (mit Rauschen) als Zielvariable

# %%
fun_reg, fun_reg_rand, fun_reg_chaos = et.evaluate_regressor(
    LinearRegression, fun, x_train_aug, x_test_aug, use_x_train_for_plot=True
)

# %% [markdown]
# $\mathit{fun}(x) = 0.1 \cdot x^2+ 2 \cdot \sin(x) - 2$

# %%
fun_reg.coef_, fun_reg.intercept_

# %%
fun_reg_rand.coef_, fun_reg_rand.intercept_

# %%
fun_reg_chaos.coef_, fun_reg_chaos.intercept_

# %%
et.train_and_plot_aug(lin, x_train, x_plot, x_test)

# %%
et.train_and_plot_aug(fun, x_train, x_plot, x_test, scale=0.0)

# %%
et.train_and_plot_aug(fun, x_train, x_plot, x_test, scale=0.5)

# %%
et.train_and_plot_aug(fun, x_train, x_plot, x_test, scale=1.5)

# %%
et.train_and_plot_aug(fun, x_train, x_plot, x_test, scale=3)


# %%
def fun2(x):
    return 2.8 * np.sin(x) + 0.3 * x + 0.08 * x**2 - 2.5


# %%
et.train_and_plot_aug(fun2, x_train, x_plot, x_test, scale=1.5)

# %%
def fun3(x):
    return np.sin(x) + 0.3 * x + 0.01 * x**3 - 2.5

# %%
et.train_and_plot_aug(fun3, x_train, x_plot, x_test, scale=0.5)


# %%
et.train_and_plot_aug(
    lambda x: np.select([x <= 6, x > 6], [-0.5, 3.5]), x_train, x_plot, x_test
)

# %%
