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


# %% making our own metric
def min_recall_precision(y_true, y_pred):
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    return min(recall, precision)


# %% without using make_scorer
def scorer_min_recall_precision(est, X, y_true, sample_weight=None):
    y_pred = est.predict(X)
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    return min(recall, precision)


# %% GridSearch
from sklearn.model_selection import GridSearchCV

grid = GridSearchCV(
    estimator=LogisticRegression(max_iter=1000),
    param_grid={"class_weight": [{0: 1, 1: v} for v in np.linspace(1, 20, 30)]},
    scoring={
        "precision": make_scorer(precision_score),
        "recall_score": make_scorer(recall_score),
        "min": make_scorer(min_recall_precision),
    },
    refit="min",
    return_train_score=True,
    cv=10,
    n_jobs=-1,
)
grid.fit(X, y, sample_weight=np.log(1 + df["Amount"]))
pd.DataFrame(grid.cv_results_)

# %% showing precision and recall scores on a plot

plt.figure(figsize=(12, 4))
results = pd.DataFrame(grid.cv_results_)
for score in ["mean_test_recall_score", "mean_test_precision", "mean_test_min"]:
    plt.plot([_[1] for _ in results["param_class_weight"]], results[score], label=score)
plt.legend()

# %% using other metrics
# precision_score: given that i predicted fraud, how accurate am i
# recall_score: what fraction of fraud cases did i identify
from sklearn.metrics import precision_score, recall_score

print("precision score:", precision_score(y, grid.predict(X)))

print("recall score:", recall_score(y, grid.predict(X)))


# %%
