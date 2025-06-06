# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>Hauspreise in Kalifornien</b>
# </div>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# <br/>

# %% [markdown]
#
# # Hauspreise in Kalifornien

# %% [markdown]
#
# ## Herunterladen und Analysieren der Daten

# %%
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

# %%
np.set_printoptions(precision=2)


# %%
california_housing = fetch_california_housing()

# %%
pprint(california_housing)

# %%
dir(california_housing)

# %%
pprint(california_housing.data)

# %%
pprint(california_housing.data[0])

# %%
print(california_housing.feature_names)

# %%
pprint(california_housing.target)

# %% [markdown]
#
# ## Erzeugen von Data Frames
#
# Für tabellarische Daten ist es oft sinnvoll Pandas Data Frames zu erzeugen, da
# diese viele Funktionen zur Analyse und zum Bearbeiten derartiger Datensätze anbieten.

# %%
simple_df = pd.DataFrame(
    data={
        "attr_1": [1, 2, 3, 4],
        "attr_2": [1.0, 2.0, 3.0, 4.0],
        "attr_3": [0.1, 0.5, 0.2, 0.6],
    }
)

# %%
simple_df

# %%
simple_df.info()

# %%
simple_df.describe()

# %%
simple_data = np.array([[1, 1.0, 0.1], [2, 2.0, 0.5], [3, 3.0, 0.2], [4, 4.0, 0.6]])

# %%
simple_columns = ["attr_1", "attr_2", "attr_3"]

# %%
simple_df2 = pd.DataFrame.from_records(data=simple_data, columns=simple_columns)

# %%
simple_df2

# %%
simple_df2.info()

# %%
simple_df2.describe()

# %%
simple_df2 == simple_df

# %%
all_data = np.concatenate(
    [california_housing.data, california_housing.target.reshape(-1, 1)], axis=1
)

# %%
all_data.shape

# %%
all_data.dtype

# %%
all_columns = [*california_housing.feature_names, "Target"]

# %%
pprint(all_columns, compact=True)
print("Length =", len(all_columns))

# %%
housing_df = pd.DataFrame.from_records(data=all_data, columns=all_columns)

# %%
housing_df

# %%
housing_df.info()

# %%
housing_df.describe()

# %%
california_housing_v2 = fetch_california_housing(as_frame=True)

# %%
california_housing_v2.frame

# %%
california_housing.frame

# %%
x, y = california_housing.data, california_housing.target

# %%
pprint(x)
pprint(y)

# %%
plt.hist(x=x[0], bins=50)
plt.show()

# %%
plt.hist(x=x[1], bins=50)
plt.show()

# %%
sns.histplot(data=x[0])
plt.show()

# %%
sns.set_theme()

# %%
sns.histplot(data=x[0])
plt.show()

# %%
sns.histplot(data=housing_df, x=all_columns[0])
plt.show()

# %%
housing_df.hist(figsize=(15, 9))
plt.show()

# %%
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.25, random_state=42
)

# %%
x_train.shape, x_test.shape

# %%
y_train.shape, y_test.shape

# %%
pprint(y[:5], compact=True)
pprint(y_train[:5], compact=True)
pprint(y_test[:5], compact=True)

# %%
train_df, test_df = train_test_split(housing_df, test_size=0.25, random_state=42)

# %%
train_df

# %%
lat_idx = all_columns.index("Latitude")
print(lat_idx)
lng_idx = all_columns.index("Longitude")
print(lng_idx)

# %%
plt.scatter(x=x[:, lng_idx], y=x[:, lat_idx])
plt.show()

# %%
plt.scatter(x=x[:, lng_idx], y=x[:, lat_idx], alpha=0.15)
plt.show()

# %%
sns.scatterplot(x=x[:, lng_idx], y=x[:, lat_idx])
plt.show()

# %%
sns.scatterplot(housing_df, x="Longitude", y="Latitude")
plt.show()

# %%
sns.scatterplot(housing_df, x="Longitude", y="Latitude", alpha=0.35)
plt.show()

# %%
housing_df.plot(kind="scatter", x="Longitude", y="Latitude", figsize=(6, 6), alpha=0.25)
plt.show()

# %%
housing_df.plot(
    kind="scatter",
    x="Longitude",
    y="Latitude",
    alpha=0.4,
    s=housing_df["Population"] / 50,
    figsize=(8, 6),
    c="Target",
    cmap="jet",
    colorbar=True,
)
plt.show()

# %%
sns.pairplot(housing_df[:1000])
plt.show()

# %% [markdown]
#
# ## Trainieren von Modellen

# %%
sgd_regressor = SGDRegressor()

# %%
sgd_regressor.fit(x_train, y_train)

# %%
sgd_pred = sgd_regressor.predict(x_test)

# %%
mean_squared_error(y_test, sgd_pred)

# %%
mean_squared_error(y_train, sgd_regressor.predict(x_train))

# %%
scaler = StandardScaler()

# %%
scaler.fit(x_train)

# %%
x_train_scaled = scaler.transform(x_train)

# %%
x_train[0]

# %%
x_train_scaled[0]

# %%
# x_train_scaled = scaler.fit_transform(x_train)

# %%
sgd_scaled_regressor = SGDRegressor()

# %%
sgd_scaled_regressor.fit(x_train_scaled, y_train)

# %%
sgd_scaled_pred = sgd_scaled_regressor.predict(scaler.transform(x_test))

# %%
mean_squared_error(y_test, sgd_scaled_pred)

# %%
mean_squared_error(y_train, sgd_scaled_regressor.predict(scaler.transform(x_train)))

# %%
tree_regressor = DecisionTreeRegressor()

# %%
tree_regressor.fit(x_train, y_train)

# %%
tree_predict = tree_regressor.predict(x_test)

# %%
mean_squared_error(y_test, tree_predict)

# %%
mean_squared_error(y_train, tree_regressor.predict(x_train))

# %%
rf_regressor = RandomForestRegressor()

# %%
rf_regressor.fit(x_train, y_train)

# %%
rf_predict = rf_regressor.predict(x_test)

# %%
mean_squared_error(y_test, rf_predict)

# %%
mean_squared_error(y_train, rf_regressor.predict(x_train))
