# %%
import pandas as pd
import matplotlib.pyplot as plt

# %%
df = pd.read_csv("./data/drawndata1.csv")
print(df.head(3))

X = df[["x", "y"]].values
y = df["z"] == "a"

y = y.values
X

# %% not using any scaler
plt.scatter(X[:, 0], X[:, 1], c=y)

# %% using StandardScaler
from sklearn.preprocessing import StandardScaler

X_new = StandardScaler().fit_transform(X)
plt.scatter(X_new[:, 0], X_new[:, 1], c=y)

# %%
from sklearn.preprocessing import QuantileTransformer

X_new = QuantileTransformer(n_quantiles=100).fit_transform(X)
plt.scatter(X_new[:, 0], X_new[:, 1], c=y)

# %%
