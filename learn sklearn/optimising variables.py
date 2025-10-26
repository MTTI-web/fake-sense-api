# %% testing
print("test")

# %% importing objects
import pandas as pd
from sklearn.datasets import fetch_california_housing as dataset
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from matplotlib.pylab import scatter

# %% creating the pipeline
X, y = dataset(return_X_y=True)

pipe = Pipeline(
    [
        ("scale", StandardScaler()),
        ("model", KNeighborsRegressor(n_neighbors=18)),
    ]
)

print(dataset().target)

# pipe.get_params()

# %% optimising variables
model = GridSearchCV(
    estimator=pipe,
    param_grid={"model__n_neighbors": range(16, 22, 1)},
    cv=10,
)
# n_neighbours = 18

model.fit(X, y)
pd.DataFrame(model.cv_results_)

# %%
pred = model.predict(X)
scatter(pred, y)

# %%
