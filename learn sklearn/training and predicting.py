# %%
from sklearn.datasets import load_diabetes as dataset

# %%
X, y = dataset(return_X_y=True)

# %%
print(y)

# %% Creating a model
from sklearn.neighbors import KNeighborsRegressor

model = KNeighborsRegressor(n_neighbors=1)


# %% Training the model
model.fit(X, y)
model_pred = model.predict(X)

# %% Creating a pipelin
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

pipe = Pipeline(
    [("scale", StandardScaler()), ("model", KNeighborsRegressor(n_neighbors=100))]
)

# %% Training the pipeline instead of the model
pipe.fit(X, y)

# %% Predicting with the model
pred = pipe.predict(X)
print("Predictions:", type(pred))

# %% Showing output for pipeline and standalone model
from matplotlib.pylab import scatter

print("Pipline prediction:")
scatter(pred, y)

# print("Model prediction: ")
# scatter(model_pred, y)

# %%
