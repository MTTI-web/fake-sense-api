# %% importing modules
import numpy as np
from sklearn.preprocessing import OneHotEncoder

# %% initializing non-numeric data
arr = np.array(["low", "low", "high", "medium"]).reshape(-1, 1)
arr

# %% transforming our data from non-numeric to numeric
encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
encoder.fit_transform(arr)

# %% testing a new non-fitted value
encoder.transform([["zero"]])

# %%
