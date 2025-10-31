# %% importing modules
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np

from helpers import emoji_to_text

df = pd.read_csv("./data/dataset.csv")
df.head(3)

X = df[["rating", "text"]]
y = df["label"] == "CG"

y = y.astype(int)

y.values

# %%
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

# This is the correct and standard way to define the preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        # Use TfidfVectorizer's built-in preprocessor argument
        ("tfidf", TfidfVectorizer(preprocessor=emoji_to_text), "text"),
        # Pass through the numeric 'rating' column
        ("numeric", "passthrough", ["rating"]),
    ],
    remainder="drop"
)


# Create the full pipeline
pipe = Pipeline(
    [
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(C=np.float64(4.641588833612782), max_iter=1000)),
        # ("classifier", RandomForestClassifier())
    ]
)


pipe.fit(X, y)
pred = pipe.predict(X)
pred

# %%
from sklearn.metrics import precision_score, recall_score

precision = precision_score(pred, y)
recall = recall_score(pred, y)

print("preicision:", precision)
print("recall:", recall)

# %%
from sklearn.model_selection import GridSearchCV

# Updated parameter grid
param_grid = {
    "classifier__C": np.logspace(-4, 3, 10),
    "classifier__class_weight": [{0: v, 1: 1.0} for v in np.linspace(0.1, 0.6, 10)],
}

# Your GridSearchCV object remains the same
grid = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    cv=5,
    n_jobs=-1
)

grid.fit(X, y)

print("Best parameters found: ", grid.best_params_)

# Display the results
results_df = pd.DataFrame(grid.cv_results_)

# #  %% PRECISION AND RECALL OF THE BEST MODEL
g_pred = grid.predict_proba(X)

# p = precision_score(g_pred, y)
# r = recall_score(g_pred, y)

# print("preicision:", p)
# print("recall:", r)

# %%
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

fpr, tpr, thresholds = roc_curve(y, g_pred[:, 1])
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], "k--")  # diagonal line for random guesses
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic")
plt.legend()
plt.show()
# g_pred

# %%
from sklearn.calibration import calibration_curve

prob_true, prob_pred = calibration_curve(y, g_pred[:, 1], n_bins=10)
plt.plot(prob_pred, prob_true, marker="o")
plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
plt.xlabel("Mean predicted probability")
plt.ylabel("Fraction of positives")
plt.title("Calibration Plot")
plt.legend()
plt.show()

# %%
import joblib

joblib.dump(grid, "model.pkl")
print("Dummy model 'model.pkl' created successfully!")

# %%
