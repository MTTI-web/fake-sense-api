# %%
import pandas as pd
import matplotlib.pyplot as plt

# %%
df = pd.read_csv("./data/drawndata2.csv")
print(df.head(3))

X = df[["x", "y"]].values
y = df["z"] == "a"

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

# %% Creating a pipeline using LogisticRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

pipe = Pipeline(
    [("scale", QuantileTransformer(n_quantiles=100)), ("model", LogisticRegression())]
)

pred = pipe.fit(X, y).predict(X)
plt.scatter(X[:, 0], X[:, 1], c=pred)

# %% Creating a pipeline using 2nd degree LogisticRegression
from sklearn.preprocessing import PolynomialFeatures

pipe2 = Pipeline(
    [
        ("scale", PolynomialFeatures()),
        ("model", LogisticRegression()),
    ]
)

pred2 = pipe2.fit(X, y).predict(X)
plt.scatter(X[:, 0], X[:, 1], c=pred2)

# %%
