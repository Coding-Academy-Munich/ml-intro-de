# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error

# %%
rng = np.random.default_rng(42)


# %%
def randomize(fun, x, scale=0.5):
    return fun(x) + rng.normal(size=x.shape, scale=scale)


# %%
def plot_graphs(
    f_y,
    x_train,
    x_test,
    reg,
    reg_rand,
    reg_chaos,
    y_train,
    y_rand_test,
    y_chaos_test,
    use_x_train_for_plot=False,
):
    if use_x_train_for_plot:
        # Set x_plot to x_train sorted along the first column
        x_plot = x_train[np.argsort(x_train[:, 0])]
    else:
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


# %% tags=["subslide", "keep"]
def print_evaluation(
    y_test, y_pred, y_rand_test, y_rand_pred, y_chaos_test, y_chaos_pred
):
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


# %% tags=["subslide", "keep"]
def evaluate_non_random_regressor(reg_type, f_y, x_train, x_test, *args, **kwargs):
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


# %% tags=["subslide", "keep"]
def evaluate_regressor(
    reg_type, f_y, x_train, x_test, use_x_train_for_plot=False, *args, **kwargs
):
    reg = reg_type(*args, **kwargs)
    reg_rand = reg_type(*args, **kwargs)
    reg_chaos = reg_type(*args, **kwargs)

    y_train = f_y(x_train[:, 0]).reshape(-1)
    y_test = f_y(x_test[:, 0]).reshape(-1)
    y_pred = reg.fit(x_train, y_train).predict(x_test)

    y_rand_train = randomize(f_y, x_train[:, 0]).reshape(-1)
    y_rand_test = randomize(f_y, x_test[:, 0]).reshape(-1)
    y_rand_pred = reg_rand.fit(x_train, y_rand_train).predict(x_test)

    y_chaos_train = randomize(f_y, x_train[:, 0], 1.5).reshape(-1)
    y_chaos_test = randomize(f_y, x_test[:, 0], 1.5).reshape(-1)
    y_chaos_pred = reg_chaos.fit(x_train, y_chaos_train).predict(x_test)

    plot_graphs(
        f_y,
        x_train,
        x_test,
        reg,
        reg_rand,
        reg_chaos,
        y_train,
        y_rand_test,
        y_chaos_test,
        use_x_train_for_plot=use_x_train_for_plot,
    )
    print_evaluation(
        y_test, y_pred, y_rand_test, y_rand_pred, y_chaos_test, y_chaos_pred
    )
    return reg, reg_rand, reg_chaos


# %%
def plot_regressions(regressors, f_y, x_train, x_test):
    y_train = f_y(x_train).reshape(-1)

    x_plot = np.linspace(0, 10, 500).reshape(-1, 1)
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(x=x_plot[:, 0], y=f_y(x_plot[:, 0]), ax=ax)
    sns.scatterplot(x=x_train[:, 0], y=y_train, ax=ax)
    for reg in regressors:
        sns.lineplot(x=x_plot[:, 0], y=reg.predict(x_plot), ax=ax)
    plt.show()


# %%
def plot_graphs_2(
    f_y, x_train, reg, reg_rand, reg_chaos, y_train, y_rand_train, y_chaos_train
):
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
def evaluate_regressor_2(reg_type, f_y, x_train, x_test, *args, **kwargs):
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

    plot_graphs_2(
        f_y,
        x_train,
        reg,
        reg_rand,
        reg_chaos,
        y_train,
        y_rand_train,
        y_chaos_train,
    )
    print_evaluation(
        y_test, y_pred, y_rand_test, y_rand_pred, y_chaos_test, y_chaos_pred
    )


# %% tags=["subslide"]
def train_and_plot_aug(f_y, x_train, x_plot, x_test, scale=0.5):
    from sklearn.linear_model import LinearRegression

    x_train_aug = np.concatenate([x_train, x_train * x_train, np.sin(x_train)], axis=1)
    y_plot = f_y(x_plot)

    f_r = lambda x: randomize(f_y, x, scale=scale)
    y_train = f_r(x_train_aug[:, 0])
    y_test = f_r(x_test)

    lr_aug = LinearRegression()  # Try with Ridge() as well...
    lr_aug.fit(x_train_aug, y_train)
    y_pred_test = lr_aug.predict(
        np.concatenate([x_test, x_test * x_test, np.sin(x_test)], axis=1)
    )
    x_plot2 = x_plot.reshape(-1, 1)
    y_pred_plot = lr_aug.predict(
        np.concatenate([x_plot2, x_plot2 * x_plot2, np.sin(x_plot2)], axis=1)
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.scatterplot(x=x_plot2[:, 0], y=y_plot, color="orange")
    sns.scatterplot(x=x_plot2[:, 0], y=y_pred_plot, color="red")
    sns.scatterplot(x=x_train_aug[:, 0], y=y_train, color="green")
    plt.show()

    mae_in = mean_absolute_error(y_test, y_pred_test)
    mse_in = mean_absolute_error(y_test, y_pred_test)
    rmse_in = np.sqrt(mse_in)

    y_nr = f_y(x_test)
    mae_true = mean_absolute_error(y_nr, y_pred_test)
    mse_true = mean_absolute_error(y_nr, y_pred_test)
    rmse_true = np.sqrt(mse_true)

    print(f"Vs. input: MAE: {mae_in:.2f}, MSE: {mse_in:.2f}, RMSE: {rmse_in:.2f}")
    print(f"True:      MAE: {mae_true:.2f}, MSE: {mse_true:.2f}, RMSE: {rmse_true:.2f}")
    print(f"Parameters: {lr_aug.coef_}, {lr_aug.intercept_}")
