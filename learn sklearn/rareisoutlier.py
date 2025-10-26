# %% importing modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, make_scorer

df = pd.read_csv("./data/creditcard.csv")[:80000]
df.head(6)

# %% Creating X and y
X = df.drop(columns=["Time", "Class", "Amount"]).values
y = df["Class"].values

print(f"Shape of X={X.shape}\nShape of y={y.shape}\nFraud Cases={y.sum()}")
print(list(df.columns.values))


# %% creating a model for predicting data
from sklearn.linear_model import LogisticRegression

mod = LogisticRegression(class_weight={0: 1, 1: 3})
mod.fit(X, y).predict(X).sum()

# %% GridSearch
from sklearn.model_selection import GridSearchCV


def outlier_precision(mod, X, y):
    preds = mod.predict(X)
    return precision_score(y, np.where(preds == -1, 1, 0))


def outlier_recall(mod, X, y):
    preds = mod.predict(X)
    return recall_score(y, np.where(preds == -1, 1, 0))


grid = GridSearchCV(
    estimator=IsolationForest(),
    param_grid={"contamination": np.linspace(0.001, 0.02, 10)},
    scoring={
        "precision": outlier_precision,
        "recall_score": outlier_recall,
    },
    refit="precision",
    return_train_score=True,
    cv=10,
    n_jobs=-1,
)
grid.fit(X, y)
pd.DataFrame(grid.cv_results_)

# %% showing precision and recall scores on a plot

plt.figure(figsize=(12, 4))
results = pd.DataFrame(grid.cv_results_)
for score in ["mean_test_recall_score", "mean_test_precision"]:
    plt.plot(results["param_contamination"], results[score], label=score)
plt.legend()

# %% example of IsolationForest
from sklearn.ensemble import IsolationForest
from collections import Counter

mod = IsolationForest().fit(X)
mod.predict(X)
Counter(mod.predict(X))

# %%
