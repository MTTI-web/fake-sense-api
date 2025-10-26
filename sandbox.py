# %% importing modules
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import emoji

df = pd.read_csv("./data/dataset.csv")
df.head(3)

X = df[["rating", "text"]]
y = df["label"] == "CG"

y = y.astype(int)

y.values


def emoji_to_text(text):
    return emoji.demojize(text)


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
    ]
)


# Create the full pipeline
pipe = Pipeline(
    [
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(C=0.1, max_iter=1000)),
    ]
)


pipe.fit(X, y)
pred = pipe.predict(X)
pred

# %% seeing vocab
v = TfidfVectorizer()
transformed = v.fit_transform(X)
v.vocabulary_

# %%
from sklearn.metrics import precision_score, recall_score

precision = precision_score(pred, y)
recall = recall_score(pred, y)

print("preicision:", precision)
print("recall:", recall)

# %%
from sklearn.model_selection import GridSearchCV

grid = GridSearchCV(
    estimator=pipe,
    param_grid={"classifier__C": np.logspace(-4, 3, 10)},
    cv=3,
)

grid.fit(X, y)

pd.DataFrame(grid.cv_results_)

# %%
g_pred = grid.predict_proba(X)
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
