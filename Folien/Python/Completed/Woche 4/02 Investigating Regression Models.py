# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>Investigating Regression Models</b>
# </div>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# <br/>

# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()

# %%
rng = np.random.default_rng(42)
x = rng.uniform(size=(150, 1), low=0.0, high=10.0)

# %%
plt.figure(figsize=(20, 1), frameon=False)
plt.yticks([], [])
plt.scatter(x, np.zeros_like(x), alpha=0.4);


# %%
def lin(x):
    return 0.85 * x - 1.5


# %%
def fun(x):
    return 2 * np.sin(x) + 0.1 * x ** 2 - 2


# %%
x_plot = np.linspace(0, 10, 500)
plt.figure(figsize=(12, 6))
sns.lineplot(x=x_plot, y=lin(x_plot))
sns.lineplot(x=x_plot, y=fun(x_plot));


# %%
def randomize(fun, x, scale=0.5):
    return fun(x) + rng.normal(size=x.shape, scale=scale)


# %%
x_train, x_test = x[:100], x[100:]

# %%
y_lin_train = lin(x_train).reshape(-1)
y_lin_test = lin(x_test).reshape(-1)
y_fun_train = fun(x_train.reshape(-1))
y_fun_test = fun(x_test).reshape(-1)

# %%
y_rand_lin_train = randomize(lin, x_train).reshape(-1)
y_rand_lin_test = randomize(lin, x_test).reshape(-1)
y_rand_fun_train = randomize(fun, x_train.reshape(-1))
y_rand_fun_test = randomize(fun, x_test).reshape(-1)

# %%
y_chaos_lin_train = randomize(lin, x_train, 1.5).reshape(-1)
y_chaos_lin_test = randomize(lin, x_test, 1.5).reshape(-1)
y_chaos_fun_train = randomize(fun, x_train, 1.5).reshape(-1)
y_chaos_fun_test = randomize(fun, x_test, 1.5).reshape(-1)

# %%
fig, ax = plt.subplots(figsize=(12, 8))
sns.lineplot(x=x_train[:, 0], y=y_lin_train, color="red", ax=ax)
sns.scatterplot(x=x_train[:, 0], y=y_rand_lin_train, ax=ax)
sns.scatterplot(x=x_train[:, 0], y=y_chaos_lin_train, ax=ax);

# %%
fig, ax = plt.subplots(figsize=(12, 8))
sns.lineplot(x=x_train[:, 0], y=y_fun_train, color="red", ax=ax)
sns.scatterplot(x=x_train[:, 0], y=y_rand_fun_train, ax=ax)
sns.scatterplot(x=x_train[:, 0], y=y_chaos_fun_train, ax=ax);

# %% [markdown]
#
#  ## Linear Regression

# %%
from sklearn.linear_model import LinearRegression

# %%
lr_lin = LinearRegression()
lr_rand_lin = LinearRegression()
lr_chaos_lin = LinearRegression()

# %%
lr_lin.fit(x_train, y_lin_train)
lr_rand_lin.fit(x_train, y_rand_lin_train)
lr_chaos_lin.fit(x_train, y_chaos_lin_train);

# %%
print("lr_lin", lr_lin.coef_, lr_lin.intercept_)
print("lr_rand_lin", lr_rand_lin.coef_, lr_rand_lin.intercept_)
print("lr_chaos_lin", lr_chaos_lin.coef_, lr_chaos_lin.intercept_)

# %%
y_lr_lin_pred = lr_lin.predict(x_test)
y_lr_rand_lin_pred = lr_rand_lin.predict(x_test)
y_lr_chaos_lin_pred = lr_chaos_lin.predict(x_test)

# %%
fig, ax = plt.subplots(figsize=(12, 8))
sns.lineplot(x=x_test[:, 0], y=y_lr_lin_pred, ax=ax)
sns.scatterplot(x=x_test[:, 0], y=y_lin_test, ax=ax)

sns.lineplot(x=x_test[:, 0], y=y_lr_rand_lin_pred, ax=ax)
sns.scatterplot(x=x_test[:, 0], y=y_rand_lin_test, ax=ax)

sns.lineplot(x=x_test[:, 0], y=y_lr_chaos_lin_pred, ax=ax)
sns.scatterplot(x=x_test[:, 0], y=y_chaos_lin_test, ax=ax);

# %%
from sklearn.metrics import mean_absolute_error, mean_squared_error

# %%
mae_lin = mean_absolute_error(y_lin_test, y_lr_lin_pred)
mae_rand_lin = mean_absolute_error(y_rand_lin_test, y_lr_rand_lin_pred)
mae_chaos_lin = mean_absolute_error(y_chaos_lin_test, y_lr_chaos_lin_pred)

mse_lin = mean_squared_error(y_lin_test, y_lr_lin_pred)
mse_rand_lin = mean_squared_error(y_rand_lin_test, y_lr_rand_lin_pred)
mse_chaos_lin = mean_squared_error(y_chaos_lin_test, y_lr_chaos_lin_pred)

rmse_lin = np.sqrt(mean_squared_error(y_lin_test, y_lr_lin_pred))
rmse_rand_lin = np.sqrt(mean_squared_error(y_rand_lin_test, y_lr_rand_lin_pred))
rmse_chaos_lin = np.sqrt(mean_squared_error(y_chaos_lin_test, y_lr_chaos_lin_pred))

print(
    "No randomness:      "
    f"MAE = {mae_lin:.2f}, MSE = {mse_lin:.2f}, RMSE = {rmse_lin:.2f}"
)
print(
    "Some randomness:    "
    f"MAE = {mae_rand_lin:.2f}, MSE = {mse_rand_lin:.2f}, RMSE = {rmse_rand_lin:.2f}"
)
print(
    "Lots of randomness: "
    f"MAE = {mae_chaos_lin:.2f}, MSE = {mse_chaos_lin:.2f}, RMSE = {rmse_chaos_lin:.2f}"
)


# %% [markdown]
#
#  ## Defining an Evaluation Function

# %%
def evaluate_non_random_regressor(reg_type, f_y, *args, **kwargs):
    reg = reg_type(*args, **kwargs)

    y_train = f_y(x_train).reshape(-1)
    y_test = f_y(x_test).reshape(-1)

    reg.fit(x_train, y_train)
    y_pred = reg.predict(x_test)

    x_plot = np.linspace(0, 10, 500).reshape(-1, 1)
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(x=x_plot[:, 0], y=reg.predict(x_plot), ax=ax)
    sns.lineplot(x=x_plot[:, 0], y=f_y(x_plot[:, 0]), ax=ax)
    sns.scatterplot(x=x_train[:, 0], y=y_train, ax=ax)
    plt.show()

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(
        "\nNo randomness:      " f"MAE = {mae:.2f}, MSE = {mse:.2f}, RMSE = {rmse:.2f}"
    )

    return reg


# %%
evaluate_non_random_regressor(LinearRegression, fun);


# %% [markdown]
#
# ### Underfitting
#
# Underfitting occurs when the model is not able to fit the training data.

# %%
def plot_graphs(f_y, reg, reg_rand, reg_chaos, y_train, y_rand_test, y_chaos_test):
    x_plot = np.linspace(0, 10, 500).reshape(-1, 1)
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.lineplot(x=x_plot[:, 0], y=reg.predict(x_plot), ax=ax)
    sns.scatterplot(x=x_train[:, 0], y=y_train, ax=ax)

    sns.lineplot(x=x_plot[:, 0], y=reg_rand.predict(x_plot), ax=ax)
    sns.scatterplot(x=x_test[:, 0], y=y_rand_test, ax=ax)

    sns.lineplot(x=x_plot[:, 0], y=reg_chaos.predict(x_plot), ax=ax)
    sns.scatterplot(x=x_test[:, 0], y=y_chaos_test, ax=ax)

    sns.lineplot(x=x_plot[:, 0], y=f_y(x_plot[:, 0]), ax=ax)
    plt.show()


# %%
def print_evaluation(y_test, y_pred, y_rand_test, y_rand_pred, y_chaos_test, y_chaos_pred):
    mae = mean_absolute_error(y_test, y_pred)
    mae_rand = mean_absolute_error(y_rand_test, y_rand_pred)
    mae_chaos = mean_absolute_error(y_chaos_test, y_chaos_pred)

    mse = mean_squared_error(y_test, y_pred)
    mse_rand = mean_squared_error(y_rand_test, y_rand_pred)
    mse_chaos = mean_squared_error(y_chaos_test, y_chaos_pred)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    rmse_rand = np.sqrt(mean_squared_error(y_rand_test, y_rand_pred))
    rmse_chaos = np.sqrt(mean_squared_error(y_chaos_test, y_chaos_pred))

    print(
        "\nNo randomness:      " f"MAE = {mae:.2f}, MSE = {mse:.2f}, RMSE = {rmse:.2f}"
    )
    print(
        "Some randomness:    "
        f"MAE = {mae_rand:.2f}, MSE = {mse_rand:.2f}, RMSE = {rmse_rand:.2f}"
    )
    print(
        "Lots of randomness: "
        f"MAE = {mae_chaos:.2f}, MSE = {mse_chaos:.2f}, RMSE = {rmse_chaos:.2f}"
    )


# %%
def evaluate_regressor(reg_type, f_y, *args, **kwargs):
    reg = reg_type(*args, **kwargs)
    reg_rand = reg_type(*args, **kwargs)
    reg_chaos = reg_type(*args, **kwargs)

    y_train = f_y(x_train).reshape(-1)
    y_test = f_y(x_test).reshape(-1)
    y_pred = reg.fit(x_train, y_train).predict(x_test)

    y_rand_train = randomize(f_y, x_train).reshape(-1)
    y_rand_test = randomize(f_y, x_test).reshape(-1)
    y_rand_pred = reg_rand.fit(x_train, y_rand_train).predict(x_test)

    y_chaos_train = randomize(f_y, x_train, 1.5).reshape(-1)
    y_chaos_test = randomize(f_y, x_test, 1.5).reshape(-1)
    y_chaos_pred = reg_chaos.fit(x_train, y_chaos_train).predict(x_test)

    plot_graphs(f_y, reg, reg_rand, reg_chaos, y_train, y_rand_test, y_chaos_test)
    print_evaluation(y_test, y_pred, y_rand_test, y_rand_pred, y_chaos_test, y_chaos_pred)


# %%
evaluate_regressor(LinearRegression, lin)

# %%
evaluate_regressor(LinearRegression, fun)

# %% [markdown]
# ## Stochastic Gradient Descent

# %%
from sklearn.linear_model import SGDRegressor

# %%
evaluate_regressor(SGDRegressor, lin)

# %%
evaluate_regressor(SGDRegressor, fun)

# %% [markdown]
# ### Unnormalized Data
#
# Stochastic gradient descent can sometimes run into trouble when different features are of different magnitudes.

# %% [markdown]
# ## Nearest Neighbors
#
# <img src="img/KnnClassification.svg" style="width: 40%; margin-left: auto; margin-right: auto;"/>

# %%
from sklearn.neighbors import KNeighborsRegressor

# %%
evaluate_non_random_regressor(KNeighborsRegressor, lin);

# %%
evaluate_non_random_regressor(KNeighborsRegressor, lin, n_neighbors=1);

# %%
evaluate_non_random_regressor(KNeighborsRegressor, lin, n_neighbors=2);

# %%
evaluate_non_random_regressor(KNeighborsRegressor, lin, n_neighbors=50);

# %%
evaluate_regressor(KNeighborsRegressor, lin)

# %%
evaluate_regressor(KNeighborsRegressor, fun)

# %% [markdown]
# ## Decision Trees

# %% [markdown]
# ## Training Decision Trees
#
# - (Conceptually) build a nested `if`-`then`-`else` statement
# - Each test compares one feature against a value
# - Minimize some *loss function* (e.g., mean squared error)
#
# ```python
# if feature_1 < 1.2:
#     if feature_2 < 3.0:
#         if feature_1 < 0.2:
#             return 25_324
#         else:
#             return 17_145
#     else:
#         ...
# else:
#     ...
# ```

# %%
x_plot = np.linspace(0, 10, 500)
plt.figure(figsize=(12, 4))
sns.lineplot(x=x_plot, y=fun(x_plot));

# %%
from sklearn.metrics import mean_squared_error
x_plot = np.linspace(0, 10, 500)
y_plot = np.ones_like(x_plot) * 1.5
print(mean_squared_error(fun(x_plot), y_plot))

# %%
plt.figure(figsize=(12, 4))
sns.lineplot(x=x_plot, y=fun(x_plot))
sns.lineplot(x=x_plot, y=y_plot);


# %%
def approx(x):
    if x < 6.2:
        return -0.5
    else:
        return 5.5


# %%
approx(4), approx(8)

# %%
list(map(approx, np.arange(4, 10)))

# %%
x_plot = np.linspace(0, 10, 500)
y_plot = list(map(approx, x_plot))
print(mean_squared_error(fun(x_plot), y_plot))

# %%
plt.figure(figsize=(12, 4))
sns.lineplot(x=x_plot, y=fun(x_plot))
sns.lineplot(x=x_plot, y=y_plot);

# %%
plt.figure(figsize=(12, 4))
sns.lineplot(x=x_plot, y=fun(x_plot));
sns.lineplot(x=x_plot, y=np.select([x_plot<6.2, x_plot>=6.2], [-0.5, 5.5]));

# %% [markdown]
# ## Advantages
#
# - Simple to understand and visualize
# - Interpretable
# - Good for analyzing the dataset (e.g., feature importances, ...)
# - Robust to statistical variations in the data
# - Needs little data preparation (not sensitive to mean or standard deviation of features, ...)

# %% [markdown]
# ## Disadvantages
#
# - Very prone to overfitting
# - Sensitive to unbalanced data sets
# - Only discrete predictions with axis-aligned boundaries
# - Unstable when training data changes (tree may change completely when a single item is added)

# %%
from sklearn.tree import DecisionTreeRegressor

# %%
evaluate_non_random_regressor(DecisionTreeRegressor, lin);

# %%
evaluate_non_random_regressor(DecisionTreeRegressor, fun);

# %%
evaluate_non_random_regressor(DecisionTreeRegressor, fun, max_depth=4);

# %%
dt1 = evaluate_non_random_regressor(DecisionTreeRegressor, fun, max_depth=1);

# %%
from sklearn.tree import plot_tree;

# %%
fig, ax = plt.subplots(figsize=(12, 4))
plot_tree(dt1, ax=ax, precision=1);

# %%
dt2 = evaluate_non_random_regressor(DecisionTreeRegressor, fun, max_depth=2);

# %%
fig, ax = plt.subplots(figsize=(16, 8))
plot_tree(dt2, ax=ax, precision=1);


# %%
def plot_regressions(regressors, f_y):
    y_train = f_y(x_train).reshape(-1)

    x_plot = np.linspace(0, 10, 500).reshape(-1, 1)
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(x=x_plot[:, 0], y=f_y(x_plot[:, 0]), ax=ax)
    sns.scatterplot(x=x_train[:, 0], y=y_train, ax=ax)
    for reg in regressors:
        sns.lineplot(x=x_plot[:, 0], y=reg.predict(x_plot), ax=ax)
    plt.show()


# %%
plot_regressions([dt1, dt2], fun);

# %%
dt3 = evaluate_non_random_regressor(DecisionTreeRegressor, fun, max_depth=3);

# %%
fig, ax = plt.subplots(figsize=(18, 10))
plot_tree(dt3, ax=ax, precision=1);

# %%
plot_regressions([dt1, dt2, dt3], fun);

# %%
dt1_mae = evaluate_non_random_regressor(
    DecisionTreeRegressor, fun, max_depth=1, criterion="absolute_error"
)

# %%
dt2_mae = evaluate_non_random_regressor(
    DecisionTreeRegressor, fun, max_depth=2, criterion="absolute_error"
)

# %%
dt3_mae = evaluate_non_random_regressor(
    DecisionTreeRegressor, fun, max_depth=3, criterion="absolute_error"
);

# %%
plot_regressions([dt1_mae, dt2_mae, dt3_mae], fun)

# %%
plot_regressions([dt1, dt1_mae], fun)

# %%
plot_regressions([dt2, dt2_mae], fun)

# %%
plot_regressions([dt3, dt3_mae], fun)


# %%
def plot_graphs(f_y, reg, reg_rand, reg_chaos, y_train, y_rand_train, y_chaos_train):
    x_plot = np.linspace(0, 10, 500).reshape(-1, 1)
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.lineplot(x=x_plot[:, 0], y=reg.predict(x_plot), ax=ax)
    sns.scatterplot(x=x_train[:, 0], y=y_train, ax=ax)

    sns.lineplot(x=x_plot[:, 0], y=reg_rand.predict(x_plot), ax=ax)
    sns.scatterplot(x=x_train[:, 0], y=y_rand_train, ax=ax)

    sns.lineplot(x=x_plot[:, 0], y=reg_chaos.predict(x_plot), ax=ax)
    sns.scatterplot(x=x_train[:, 0], y=y_chaos_train, ax=ax)

    sns.lineplot(x=x_plot[:, 0], y=f_y(x_plot[:, 0]), ax=ax)
    plt.show()


# %%
def evaluate_regressor(reg_type, f_y, *args, **kwargs):
    reg = reg_type(*args, **kwargs)
    reg_rand = reg_type(*args, **kwargs)
    reg_chaos = reg_type(*args, **kwargs)

    y_train = f_y(x_train).reshape(-1)
    y_test = f_y(x_test).reshape(-1)
    y_pred = reg.fit(x_train, y_train).predict(x_test)

    y_rand_train = randomize(f_y, x_train).reshape(-1)
    y_rand_test = randomize(f_y, x_test).reshape(-1)
    y_rand_pred = reg_rand.fit(x_train, y_rand_train).predict(x_test)

    y_chaos_train = randomize(f_y, x_train, 1.5).reshape(-1)
    y_chaos_test = randomize(f_y, x_test, 1.5).reshape(-1)
    y_chaos_pred = reg_chaos.fit(x_train, y_chaos_train).predict(x_test)

    plot_graphs(f_y, reg, reg_rand, reg_chaos, y_train, y_rand_train, y_chaos_train)
    print_evaluation(y_test, y_pred, y_rand_test, y_rand_pred, y_chaos_test, y_chaos_pred)


# %%
evaluate_regressor(DecisionTreeRegressor, lin)

# %%
evaluate_regressor(DecisionTreeRegressor, lin, max_depth=3)

# %% [markdown]
# # Overfitting and Underfitting
#
# Overfitting occurs when the model matches noise in the training data.
#
# Underfitting occurs when the model cannot match the training data set.

# %% [markdown]
# ## Bias/Variance Tradeoff
#
# **Bias error:** Error from bad assumptions in our model or learning algorithm about the structure of our solution. Leads to underfitting.
#
# **Variance:** Reaction to small fluctuations in the training data. Leads to overfitting.
#
# "\[B\]ias measures the tendency of a system to consistently learn the  wrong things, and variance measures its tendency to learn irrelevant  details"
#
# -- Pedro Domingos
#

# %% [markdown]
# ### Bias/Variance Tradeoff
#
# <img src="img/low-bias-low-variance.png" style="width: 22%; display: inline-block;"/>
# <img src="img/low-bias-high-variance.png" style="width: 22%; display: inline-block;"/>
# <img src="img/high-bias-low-variance.png" style="width: 22%; display: inline-block;"/>
# <img src="img/high-bias-high-variance.png" style="width: 22%; display: inline-block;"/>

# %% [markdown]
# <img src="img/Figure-09-008.png" style="width: 40%; padding: 20px;"/>

# %% [markdown]
# <img src="img/Figure-09-009.png" style="width: 80%; padding: 20px;"/>

# %% [markdown]
# <img src="img/Figure-09-010.png" style="width: 40%; padding: 20px;"/>

# %%
x = np.linspace(0, 10, 500)
y = randomize(fun, x, 2.0)
lr_reg = LinearRegression().fit(x.reshape(-1, 1), y)
dt_reg = DecisionTreeRegressor().fit(x.reshape(-1, 1), y)

def plot(reg, ax):
    sns.lineplot(x=x, y=fun(x), ax=ax, color="red")
    sns.lineplot(x=x, y=y, ax=ax, alpha=0.5)
    sns.lineplot(x=x, y=reg.predict(x.reshape(-1, 1)), ax=ax);


# %%
fig, ax = plt.subplots(ncols=2, figsize=(20, 6))
plot(lr_reg, ax[0])
plot(dt_reg, ax[1])

# %%
x = np.linspace(0, 10, 500)
fix, ax = plt.subplots(figsize=(22, 12))
# ax.fill_between(x, 1.8 * np.sin(x) + 0.09 * x ** 2 - 5, 2.1 * np.sin(x) + 0.11 * x ** 2 + 1, alpha=0.2)
ax.plot(x, randomize(fun, x, scale=0.5))
ax.plot(x, fun(x), color="r");


# %%
def fun2(x): return 2.8 * np.sin(x) + 0.3 * x + 0.08 * x ** 2 - 2.5
fix, ax = plt.subplots(figsize=(22, 12))
# ax.fill_between(x, 1.8 * np.sin(x) + 0.09 * x ** 2 - 5, 2.1 * np.sin(x) + 0.11 * x ** 2 + 1, alpha=0.2)
ax.plot(x, randomize(fun2, x, scale=0.4), color="orange")
ax.plot(x, fun2(x), color="yellow")
ax.plot(x, fun(x), color="r");

# %%
fix, ax = plt.subplots(figsize=(22, 12))
ax.fill_between(x, 1.8 * np.sin(x) + 0.09 * x ** 2 - 5, 2.1 * np.sin(x) + 0.11 * x ** 2 + 1, alpha=0.2)
ax.plot(x, randomize(fun2, x, scale=0.4), color="orange")
# ax.plot(x, randomize(fun, x, scale=0.5))
ax.plot(x, fun2(x), color="yellow")
ax.plot(x, fun(x), color="r")
ax.plot(x, np.select([x<=6, x>6], [-0.5, 3.5]), color="darkgreen");

# %% [markdown]
# ## Recognizing Overfitting
#
# - Performance on your training set is much better than on your validation/test set

# %% [markdown]
# ## Reducing Overfitting
#
# - Collect more/different training data
# - Perform (better) feature engineering
# - Decrease model capacity
# - Regularize the model
# - Use cross validation
# - Augment the training data
# - Add batch normalization, dropout, ... layers
# - Stop training early
# - ...

# %%
rng = np.random.default_rng(42)
dt_reg = DecisionTreeRegressor()
x_train = rng.uniform(size=(100, 1), low=0.0, high=10.0)
y_train = randomize(fun, x_train, scale=1.5)
y_test = randomize(fun, x_train, scale=1.5)
dt_reg.fit(x_train, y_train)

# %%
mae = mean_absolute_error(y_train, dt_reg.predict(x_train))
mse = mean_squared_error(y_train, dt_reg.predict(x_train))
mae, mse

# %%
mae_test = mean_absolute_error(y_test, dt_reg.predict(x_train))
mse_test = mean_squared_error(y_test, dt_reg.predict(x_train))
mae_test, mse_test

# %%
fun_mae = mean_absolute_error(y_train, fun(x_train))
fun_mse = mean_squared_error(y_train, fun(x_train))
fun_mae, fun_mse

# %%
fig, ax = plt.subplots(figsize=(16, 6))
sns.lineplot(x=x_train[:, 0], y=fun(x_train)[:, 0])
sns.scatterplot(x=x_train[:, 0], y=y_train[:, 0])
sns.lineplot(x=x_plot, y=dt_reg.predict(x_plot.reshape(-1, 1)));

# %%
fig, ax = plt.subplots(figsize=(16, 6))
sns.lineplot(x=x_train[:, 0], y=fun(x_train)[:, 0])
sns.scatterplot(x=x_train[:, 0], y=y_test[:, 0])
sns.lineplot(x=x_plot, y=dt_reg.predict(x_plot.reshape(-1, 1)));

# %%
evaluate_regressor(DecisionTreeRegressor, fun)

# %%
evaluate_regressor(DecisionTreeRegressor, fun, max_depth=2)

# %%
evaluate_regressor(DecisionTreeRegressor, fun, max_depth=3)

# %%
evaluate_regressor(DecisionTreeRegressor, fun, max_depth=2, criterion="absolute_error")

# %%
# evaluate_regressor(DecisionTreeRegressor, fun, max_leaf_nodes=20)
evaluate_regressor(DecisionTreeRegressor, fun, max_leaf_nodes=20, criterion="absolute_error")

# %%
evaluate_regressor(DecisionTreeRegressor, fun, min_samples_split=16)

# %%
evaluate_regressor(DecisionTreeRegressor, fun, min_samples_leaf=8)

# %%
