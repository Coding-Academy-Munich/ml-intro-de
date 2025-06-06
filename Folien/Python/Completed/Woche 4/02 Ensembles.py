# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>Ensembles</b>
# </div>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# <br/>

# %% [markdown]
#
# # Ensembles

# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
import evaluation_tools as et

# %%
sns.set_theme()

# %%
rng = np.random.default_rng(42)
x = rng.uniform(size=(150, 1), low=0.0, high=10.0)
x_train, x_test = x[:100], x[100:]
x_plot = np.linspace(0, 10, 500).reshape(-1, 1)


# %%
def lin(x):
    return 0.85 * x - 1.5


# %%
def fun(x):
    return 2 * np.sin(x) + 0.1 * x**2 - 2


# %% [markdown]
# # Ensembles, Random Forests, Gradient Boosted Trees

# %% [markdown]
#
# ## Ensemble Methoden
#
# Idee: Kombiniere mehrere Schätzungen, um die Gesamtleistung zu verbessern.
#
# - Averaging-Methoden:
#   - Unabhängige Schätzer (Estimators), durchschnittliche Vorhersagen
#   - Reduziert die Varianz (overfitting)
#   - Bagging, Random Forests
# - Boosting methoden:
#   - Schätzer sequenziell trainieren
#   – Jeder Schätzer wird darauf trainiert, die Verzerrung seiner (kombinierten)
#     Vorgänger zu reduzieren

# %% [markdown]
# ### Bagging
#
# - Durchschnitts-Methode: Erstelle mehrere Schätzer des gleichen Typs, bilde den
#   Durchschnitt ihrer Ergebnisse
# - Benötigt eine Möglichkeit, Unterschiede zwischen Schätzern einzuführen
#   - Andernfalls wird die Varianz nicht reduziert
#   - Trainiere mit zufälligen Teilmengen der Trainingsdaten
# - Reduziert Overfitting
# - Arbeitet am besten mit starken Schätzern (z.B. Entscheidungsbäumen mit (mäßig)
#   großer Tiefe)

# %% [markdown]
# ### Random Forests
#
# - Bagging classifier/regressor, der Entscheidungsbäume verwendet
# - Für jeden Baum im Wald:
#   - Teilmenge von Trainingsdaten
#   - Teilmenge von Merkmalen
# - Oft signifikante Verringerung der Varianz (Overfitting)
# - Manchmal Erhöhung des Bias

# %%
from sklearn.ensemble import RandomForestRegressor

# %%
et.evaluate_non_random_regressor(
    RandomForestRegressor, lin, x_train, x_test, random_state=42
)

# %%
et.evaluate_non_random_regressor(
    RandomForestRegressor, fun, x_train, x_test, random_state=42
)

# %%
et.evaluate_non_random_regressor(
    RandomForestRegressor,
    fun,
    x_train,
    x_test,
    n_estimators=25,
    criterion="absolute_error",
    random_state=42,
)

# %%
et.evaluate_regressor(RandomForestRegressor, lin, x_train, x_test, random_state=42)

# %%
et.evaluate_regressor(
    RandomForestRegressor,
    lin,
    x_train,
    x_test,
    n_estimators=500,
    max_depth=3,
    random_state=42,
)

# %%
et.evaluate_regressor(
    RandomForestRegressor,
    lin,
    x_train,
    x_test,
    n_estimators=500,
    min_samples_leaf=6,
    random_state=42,
)

# %%
et.evaluate_regressor(RandomForestRegressor, fun, x_train, x_test, random_state=42)

# %%
et.evaluate_regressor(
    RandomForestRegressor,
    fun,
    x_train,
    x_test,
    n_estimators=1000,
    min_samples_leaf=6,
    random_state=43,
    n_jobs=-1,
)

# %% [markdown]
# ## Gradient Boosted Trees
#
# - Boosting-Methode sowohl für die Regression als auch für die Klassifizierung
# - Benötigt differenzierbare Verlustfunktion

# %%
from sklearn.ensemble import GradientBoostingRegressor

# %%
et.evaluate_non_random_regressor(GradientBoostingRegressor, lin, x_train, x_test)

# %%
et.evaluate_non_random_regressor(GradientBoostingRegressor, fun, x_train, x_test)

# %%
et.evaluate_regressor(GradientBoostingRegressor, lin, x_train, x_test)

# %%
et.evaluate_regressor(
    GradientBoostingRegressor,
    lin,
    x_train,
    x_test,
    n_estimators=200,
    learning_rate=0.05,
    loss="absolute_error",
)

# %%
et.evaluate_regressor(
    GradientBoostingRegressor,
    lin,
    x_train,
    x_test,
    n_estimators=500,
    learning_rate=0.01,
    loss="absolute_error",
    subsample=0.1,
    random_state=46,
)

# %%
et.evaluate_regressor(
    GradientBoostingRegressor,
    fun,
    x_train,
    x_test,
    n_estimators=500,
    learning_rate=0.01,
    loss="absolute_error",
    subsample=0.1,
    random_state=44,
)

# %% [markdown]
# ## Mehrere Features

# %%
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

np.set_printoptions(precision=1)

# %%
x, y, coef = make_regression(
    n_samples=250, n_features=4, n_informative=1, coef=True, random_state=42
)
x.shape, y.shape, coef

# %%
fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(20, 12))
for i, ax in enumerate(axs.reshape(-1)):
    sns.scatterplot(x=x[:, i], y=y, ax=ax)
plt.show()

# %%
x, y, coef = make_regression(
    n_samples=250, n_features=20, n_informative=10, coef=True, random_state=42
)
x.shape, y.shape, coef

# %%
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4)
x_train.shape, x_test.shape, y_train.shape, y_test.shape

# %%
fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(20, 12))
for i in range(2):
    sns.scatterplot(x=x[:, i], y=y, ax=axs[0, i])
for i in range(2):
    sns.scatterplot(x=x[:, i + 6], y=y, ax=axs[1, i])
plt.show()

# %%
lr_clf = LinearRegression()
lr_clf.fit(x_train, y_train)
y_lr_pred = lr_clf.predict(x_test)

mean_absolute_error(y_test, y_lr_pred), mean_squared_error(y_test, y_lr_pred)

# %%
lr_clf.coef_.astype(np.int32), coef.astype(np.int32)

# %%
dt_clf = DecisionTreeRegressor()
dt_clf.fit(x_train, y_train)
y_dt_pred = dt_clf.predict(x_test)

mean_absolute_error(y_test, y_dt_pred), mean_squared_error(y_test, y_dt_pred)

# %%
rf_clf = RandomForestRegressor()
rf_clf.fit(x_train, y_train)
y_rf_pred = rf_clf.predict(x_test)

mean_absolute_error(y_test, y_rf_pred), mean_squared_error(y_test, y_rf_pred)

# %%
gb_clf = GradientBoostingRegressor()
gb_clf.fit(x_train, y_train)
y_gb_pred = gb_clf.predict(x_test)

mean_absolute_error(y_test, y_gb_pred), mean_squared_error(y_test, y_gb_pred)

# %%
x, y, coef = make_regression(
    n_samples=250,
    n_features=20,
    n_informative=10,
    noise=100.0,
    coef=True,
    random_state=42,
)
x.shape, y.shape, coef

# %%
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4)
x_train.shape, x_test.shape, y_train.shape, y_test.shape

# %%
lr_clf = LinearRegression()
lr_clf.fit(x_train, y_train)
y_lr_pred = lr_clf.predict(x_test)

mean_absolute_error(y_test, y_lr_pred), mean_squared_error(y_test, y_lr_pred)

# %%
dt_clf = DecisionTreeRegressor()
dt_clf.fit(x_train, y_train)
y_dt_pred = dt_clf.predict(x_test)

mean_absolute_error(y_test, y_dt_pred), mean_squared_error(y_test, y_dt_pred)

# %%
rf_clf = RandomForestRegressor()
rf_clf.fit(x_train, y_train)
y_rf_pred = rf_clf.predict(x_test)

mean_absolute_error(y_test, y_rf_pred), mean_squared_error(y_test, y_rf_pred)

# %%
gb_clf = GradientBoostingRegressor()
gb_clf.fit(x_train, y_train)
y_gb_pred = gb_clf.predict(x_test)

mean_absolute_error(y_test, y_gb_pred), mean_squared_error(y_test, y_gb_pred)

# %%
x, y, coef = make_regression(
    n_samples=250,
    n_features=20,
    n_informative=10,
    noise=100.0,
    coef=True,
    random_state=42,
)
y += (20 * x[:, 1]) ** 2
x.shape, y.shape, coef

# %%
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4)
x_train.shape, x_test.shape, y_train.shape, y_test.shape

# %%
fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(20, 12))
for i in range(2):
    sns.scatterplot(x=x[:, i], y=y, ax=axs[0, i])
for i in range(2):
    sns.scatterplot(x=x[:, i + 6], y=y, ax=axs[1, i])
plt.show()

# %%
lr_clf = LinearRegression()
lr_clf.fit(x_train, y_train)
y_lr_pred = lr_clf.predict(x_test)

mean_absolute_error(y_test, y_lr_pred), mean_squared_error(y_test, y_lr_pred)

# %%
dt_clf = DecisionTreeRegressor()
dt_clf.fit(x_train, y_train)
y_dt_pred = dt_clf.predict(x_test)

mean_absolute_error(y_test, y_dt_pred), mean_squared_error(y_test, y_dt_pred)

# %%
rf_clf = RandomForestRegressor()
rf_clf.fit(x_train, y_train)
y_rf_pred = rf_clf.predict(x_test)

mean_absolute_error(y_test, y_rf_pred), mean_squared_error(y_test, y_rf_pred)

# %%
gb_clf = GradientBoostingRegressor()
gb_clf.fit(x_train, y_train)
y_gb_pred = gb_clf.predict(x_test)

mean_absolute_error(y_test, y_gb_pred), mean_squared_error(y_test, y_gb_pred)
