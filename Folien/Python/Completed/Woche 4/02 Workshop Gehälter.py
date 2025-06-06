# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>Workshop Gehälter</b>
# </div>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# <br/>

# %% [markdown]
#
# The module `fake_salary` contains a number of synthetic datasets that
# represent salary as a function of ages and education level, or ages and
# profession.
#
# Analyze how `linear_salaries`, `stepwise_salaries`, `interpolated_salaries`
# and `multivar_salaries` depend on `ages` and `education_levels` and train
# regression models (at least linear and decision tree models) that model these
# dependencies.
#
# Do the same for `multidist_ages`, `professions`, and `multidist_salaries`.
#
# *Hint:* The `fake_salary` module contains a number of plots that show the
# relatinships; to display them run the file as main module or interactively in
# VS Code. Please try to solve the exercises yourself before looking at the
# plots.

# %%
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

# %%
from fake_salary import (
    ages,
    education_levels,

    linear_salaries,
    stepwise_salaries,
    interpolated_salaries,
    multivar_salaries,

    multidist_ages,
    professions,

    multidist_salaries,
)

# %%
ages.shape, education_levels.shape

# %%
(linear_salaries.shape,
 stepwise_salaries.shape,
 interpolated_salaries.shape,
 multivar_salaries.shape)

# %%
sns.set_theme(style="darkgrid")

# %%
sns.scatterplot(x=ages[:500], y=linear_salaries[:500]);

# %%
sns.scatterplot(x=ages, y=linear_salaries, alpha=0.15);

# %%
sns.regplot(x=ages[:500], y=linear_salaries[:500], line_kws={"color": "red"})

# %%
# Salaries approximately taken from
# https://www.indeed.com/career-advice/pay-salary/average-salary-by-age


# %%
sns.scatterplot(x=ages[:500], y=interpolated_salaries[:500]);


# %%
sns.scatterplot(x=ages, y=interpolated_salaries, alpha=0.15);


# %%
sns.scatterplot(x=ages, y=linear_salaries, alpha=0.15)
sns.scatterplot(x=ages, y=stepwise_salaries, alpha=0.15)
sns.scatterplot(x=ages, y=interpolated_salaries, alpha=0.15)

# %%
linear_salaries_df = pd.DataFrame(
    {"age": np.round(ages), "salary": np.round(linear_salaries)}
)
stepwise_salaries_df = pd.DataFrame(
    {"age": np.round(ages), "salary": np.round(stepwise_salaries)}
)
interpolated_salaries_df = pd.DataFrame(
    {"age": np.round(ages), "salary": np.round(interpolated_salaries)}
)

# %%
linear_salaries_df

# %%
sns.scatterplot(data=linear_salaries_df, x="age", y="salary", alpha=0.25)
sns.scatterplot(data=stepwise_salaries_df, x="age", y="salary", alpha=0.25)
sns.scatterplot(data=interpolated_salaries_df, x="age", y="salary", alpha=0.25);

# %%
sns.scatterplot(x=ages, y=multivar_salaries, hue=education_levels)

# %%
multivar_salaries_df = pd.DataFrame(
    {"age": np.round(ages), "edu_lvl": education_levels, "salary": multivar_salaries}
)

# %%
sns.scatterplot(
    data=multivar_salaries_df, x="age", y="salary", hue="edu_lvl", alpha=0.25
)

# %%
sns.pairplot(data=multivar_salaries_df);

# %%
grid = sns.pairplot(
    data=multivar_salaries_df,
    vars=["age", "salary", "edu_lvl"],
    hue="edu_lvl",
    diag_kind="hist",
    height=3,
    aspect=1,
)

# %%
sns.scatterplot(x=multidist_ages, y=multidist_salaries, alpha=0.15);

# %%
fig, ax = plt.subplots(figsize=(9, 8))
sns.scatterplot(
    x=multidist_ages,
    y=multidist_salaries,
    hue=professions,
    style=professions,
    # palette="flare",
    ax=ax,
    alpha=0.5,
);

# %%
fig, ax = plt.subplots(figsize=(9, 8))
sns.scatterplot(
    x=multidist_ages,
    y=multidist_salaries,
    hue=professions,
    # style=professions,
    # palette="coolwarm",
    # palette="seismic",
    palette="gist_rainbow",
    ax=ax,
    alpha=0.5,
)


# %%
multidist_salaries_df = pd.DataFrame(
    {
        "age": np.round(multidist_ages),
        "profession": professions,
        "salary": np.round(multidist_salaries),
    }
)

# %%
fig, ax = plt.subplots(figsize=(9, 8))
sns.scatterplot(
    data=multidist_salaries_df,
    x="age",
    y="salary",
    hue="profession",
    palette="gist_rainbow",
    ax=ax,
    alpha=0.5,
)


# %%
fig, axes = plt.subplots(
    ncols=2,
    nrows=3,
    figsize=(9, 9),
    # sharex=True,
    sharey=True,
    gridspec_kw={"hspace": 0.5, "wspace": 0.2},
)
for row in range(3):
    for col in range(2):
        profession = col + 2 * row
        ax = axes[row, col]
        ax.set_title(f"profession={profession}")
        sns.scatterplot(
            data=multidist_salaries_df.loc[
                multidist_salaries_df["profession"] == profession
            ],
            x="age",
            y="salary",
            ax=ax,
            alpha=0.25,
        )

# %%
grid = sns.pairplot(
    data=multidist_salaries_df, hue="profession", height=3, aspect=1.5
)

# %%
X1 = ages.reshape(-1, 1)

# %%
X1_lin_train, X1_lin_test, y1_lin_train, y1_lin_test = train_test_split(X1, linear_salaries, test_size=0.2)

# %%
X1_lin_train.shape, X1_lin_test.shape, y1_lin_train.shape, y1_lin_test.shape

# %%
sgd1_lin = SGDRegressor().fit(X1_lin_train, y1_lin_train)


# %%
def evaluate_model(model, X_train, X_test, y_train, y_test):
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    mae_train = mean_absolute_error(y_train, y_train_pred)
    mae_test = mean_absolute_error(y_test, y_test_pred)
    rmse_train = root_mean_squared_error(y_train, y_train_pred)
    rmse_test = root_mean_squared_error(y_test, y_test_pred)
    print(f"MAE train:  {mae_train:.2f}, test: {mae_test:.2f}")
    print(f"RMSE train: {rmse_train:.2f}, test: {rmse_test:.2f}")


# %%
evaluate_model(sgd1_lin, X1_lin_train, X1_lin_test, y1_lin_train, y1_lin_test)


# %%
def plot_results(model, X_train, X_test, y_train, y_test):
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    plt.figure(figsize=(9, 6))
    plt.scatter(X_train[:, 0], y_train, alpha=0.25, label="train")
    plt.scatter(X_test[:, 0], y_test, alpha=0.25, label="test")
    plt.scatter(X_train[:, 0], y_train_pred, alpha=0.25, label="train pred")
    plt.scatter(X_test[:, 0], y_test_pred, alpha=0.25, label="test pred")
    plt.legend()
    plt.show()


# %%
plot_results(sgd1_lin, X1_lin_train, X1_lin_test, y1_lin_train, y1_lin_test)

# %%
X = np.concatenate([ages.reshape(-1, 1), education_levels.reshape(-1, 1)], axis=1)
X.shape

# %%
X_lin_train, X_lin_test, y_lin_train, y_lin_test = train_test_split(X, linear_salaries, test_size=0.2)

# %%
X_lin_train.shape, X_lin_test.shape, y_lin_train.shape, y_lin_test.shape

# %%
sgd_lin = SGDRegressor().fit(X_lin_train, y_lin_train)

# %%
evaluate_model(sgd_lin, X_lin_train, X_lin_test, y_lin_train, y_lin_test)

# %%
plot_results(sgd_lin, X_lin_train, X_lin_test, y_lin_train, y_lin_test)

# %%
rf_lin = RandomForestRegressor(max_depth=6).fit(X_lin_train, y_lin_train)

# %%
evaluate_model(rf_lin, X_lin_train, X_lin_test, y_lin_train, y_lin_test)

# %%
plot_results(rf_lin, X_lin_train, X_lin_test, y_lin_train, y_lin_test)

# %%
X_int_train, X_int_test, y_int_train, y_int_test = train_test_split(X, interpolated_salaries, test_size=0.2)

# %%
sgd_int = SGDRegressor().fit(X_int_train, y_int_train)

# %%
evaluate_model(sgd_int, X_int_train, X_int_test, y_int_train, y_int_test)

# %%
plot_results(sgd_int, X_int_train, X_int_test, y_int_train, y_int_test)

# %%
rf_int = RandomForestRegressor(max_depth=6).fit(X_int_train, y_int_train)

# %%
evaluate_model(rf_int, X_int_train, X_int_test, y_int_train, y_int_test)

# %%
plot_results(rf_int, X_int_train, X_int_test, y_int_train, y_int_test)

# %%
X_sw_train, X_sw_test, y_sw_train, y_sw_test = train_test_split(X, stepwise_salaries, test_size=0.2)

# %%
sgd_sw = SGDRegressor().fit(X_sw_train, y_sw_train)

# %%
evaluate_model(sgd_sw, X_sw_train, X_sw_test, y_sw_train, y_sw_test)

# %%
plot_results(sgd_sw, X_sw_train, X_sw_test, y_sw_train, y_sw_test)

# %%
rf_sw = RandomForestRegressor(max_depth=6).fit(X_sw_train, y_sw_train)

# %%
evaluate_model(rf_sw, X_sw_train, X_sw_test, y_sw_train, y_sw_test)

# %%
plot_results(rf_sw, X_sw_train, X_sw_test, y_sw_train, y_sw_test)

# %%
X_mv_train, X_mv_test, y_mv_train, y_mv_test = train_test_split(X, multivar_salaries, test_size=0.2)

# %%
sgd_mv = SGDRegressor().fit(X_mv_train, y_mv_train)

# %%
evaluate_model(sgd_mv, X_mv_train, X_mv_test, y_mv_train, y_mv_test)

# %%
plot_results(sgd_mv, X_mv_train, X_mv_test, y_mv_train, y_mv_test)

# %%
rf_mv = RandomForestRegressor(max_depth=8).fit(X_mv_train, y_mv_train)

# %%
evaluate_model(rf_mv, X_mv_train, X_mv_test, y_mv_train, y_mv_test)

# %%
plot_results(rf_mv, X_mv_train, X_mv_test, y_mv_train, y_mv_test)

# %%
