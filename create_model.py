# %%
# -- coding: utf-8 --
# Minimal script to train, calibrate, and persist a calibrated LogisticRegression model.
# Produces: model_logreg_calibrated.pkl (contains {'model', 'threshold', 'best_params'}).

# %%
import numpy as np
import pandas as pd
import joblib

# %%
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import precision_recall_curve

# %%
from helpers import emoji_to_text


# %%
# Load data
df = pd.read_csv("./data/dataset.csv")
X_full = df[["rating", "text"]]
y_full = (df["label"] == "CG").astype(int)

# %%
# Split: train / calibration / test
X_train, X_temp, y_train, y_temp = train_test_split(
    X_full, y_full, test_size=0.4, stratify=y_full, random_state=42
)
X_calib, X_test, y_calib, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
)

# %%
# Preprocessor and pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("tfidf", TfidfVectorizer(preprocessor=emoji_to_text), "text"),
        ("numeric", "passthrough", ["rating"]),
    ],
    remainder="drop",
)

pipe = Pipeline(
    [
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=5000, random_state=42)),
    ]
)

# %%
# Hyperparameter tuning on TRAIN ONLY
param_grid_lr = [
    {
        "classifier__solver": ["liblinear"],
        "classifier__penalty": ["l1", "l2"],
        "classifier__C": np.logspace(-3, 2, 8),
    },
    {
        "classifier__solver": ["saga"],
        "classifier__penalty": ["l1", "l2"],
        "classifier__C": np.logspace(-3, 2, 8),
    },
]

grid = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid_lr,
    scoring="average_precision",
    cv=5,
    n_jobs=-1,
    refit=True,
)
grid.fit(X_train, y_train)
best_lr_pipe = grid.best_estimator_

# %%
# Calibrate on CALIBRATION split (prefit flow)
calibrator = CalibratedClassifierCV(
    estimator=best_lr_pipe, method="sigmoid", cv="prefit"
)
calibrator.fit(X_calib, y_calib)

# %%
# Select threshold on CALIBRATION via PR curve (target precision example)
proba_calib = calibrator.predict_proba(X_calib)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_calib, proba_calib)
target_precision = 0.90
idx = np.where(precision[:-1] >= target_precision)[0]
operating_threshold = float(thresholds[idx[0]] if len(idx) else 0.5)

# %%
# Persist single artifact
artifact = {
    "model": calibrator,  # Calibrated, ready for predict_proba
    "threshold": operating_threshold,  # Operating point for classification
    "best_params": grid.best_params_,  # For reproducibility
}
joblib.dump(artifact, "model_logreg_calibrated.pkl")

# %%
