# %%
# -- coding: utf-8 --
# Minimal script to train, calibrate, and persist a calibrated LogisticRegression model.
# Produces: model_logreg_calibrated.pkl (contains {'model', 'threshold', 'best_params'}).

# %%
import numpy as np
import pandas as pd
import joblib

# --- ADDED ---
import re  # For finding words (Regular Expressions)
from sklearn.base import BaseEstimator, TransformerMixin  # For creating a custom transformer
# --- END ADDED ---

# %%
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
import re

# %%
from helpers import emoji_to_text


# --- ADDED ---
# This is our new class to calculate the "Unique Word Ratio" feature.
# It checks for repetition, regardless of what the words are.
class RepetitionFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Creates a feature that is the ratio of unique words to total words.
    A low ratio (e.g., 0.2) indicates high repetition (e.g., "newword newword newword").
    A high ratio (e.g., 1.0) indicates no repetition.
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        # We don't need to "learn" anything, so we just return self
        return self

    def _get_unique_word_ratio(self, text):
        """Calculates the unique word ratio for a single string."""
        # Find all "words" (sequences of letters/numbers)
        words = re.findall(r'\b\w+\b', (text or "").lower())
        
        # If the review is empty or has no words, return a neutral score
        if not words:
            return 0.5  # Using 0.5 as a neutral "middle ground"
        
        total_words = len(words)
        unique_words = len(set(words))
        
        # This is our new feature:
        return unique_words / total_words

    def transform(self, X_series, y=None):
        """
        Applies the ratio calculation to every review in the 'text' column.
        The output must be a 2D numpy array for scikit-learn.
        """
        # 'X_series' will be the 'text' column from the DataFrame
        return X_series.apply(self._get_unique_word_ratio).values.reshape(-1, 1)

# --- END ADDED ---


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
        # This transformer learns the MEANING of KNOWN words
        ("tfidf", TfidfVectorizer(preprocessor=emoji_to_text), "text"),
        
        # --- ADDED ---
        # This transformer learns the STRUCTURE (repetition) of ALL words
        ("repetition_feature", RepetitionFeatureExtractor(), "text"),
        # --- END ADDED ---
        
        # This transformer just passes through the rating number
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
# --- APPEND: plotting/metrics imports ---
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
from sklearn.metrics import precision_recall_curve, precision_score, recall_score
from sklearn.calibration import calibration_curve
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# %%
# --- APPEND: ROC curve (test split) ---
# Probabilities already available via fitted models
y_test_prob_uncal = best_lr_pipe.predict_proba(X_test)[:, 1]
y_test_prob_cal   = calibrator.predict_proba(X_test)[:, 1]

# Compute ROC points
fpr_uncal, tpr_uncal, _ = roc_curve(y_test, y_test_prob_uncal)
fpr_cal,   tpr_cal,   _ = roc_curve(y_test, y_test_prob_cal)

# AUC
auc_uncal = roc_auc_score(y_test, y_test_prob_uncal)
auc_cal   = roc_auc_score(y_test, y_test_prob_cal)

# Plot
plt.figure(figsize=(8, 6))
plt.plot(fpr_uncal, tpr_uncal, label=f"Uncalibrated (AUC={auc_uncal:.3f})", lw=2)
plt.plot(fpr_cal,   tpr_cal,   label=f"Calibrated (AUC={auc_cal:.3f})",   lw=2)
plt.plot([0, 1], [0, 1], "k--", lw=1, label="Chance")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (test split)")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("roc_curve_test_manual.png")
plt.close()
print(f"Saved: roc_curve_test_manual.png (AUC uncal={auc_uncal:.4f}, cal={auc_cal:.4f})")

# %%
# --- APPEND: Calibration (reliability) curve (test split) ---
prob_true_uncal, prob_pred_uncal = calibration_curve(y_test, y_test_prob_uncal, n_bins=10, strategy="quantile")
prob_true_cal,   prob_pred_cal   = calibration_curve(y_test, y_test_prob_cal,   n_bins=10, strategy="quantile")

plt.figure(figsize=(8, 6))
# Perfect calibration
plt.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
# Curves
plt.plot(prob_pred_uncal, prob_true_uncal, marker="o", label="Uncalibrated", lw=2)
plt.plot(prob_pred_cal,   prob_true_cal,   marker="o", label="Calibrated",   lw=2)
plt.xlabel("Mean predicted probability")
plt.ylabel("Fraction of positives")
plt.title("Calibration curve (test split)")
plt.legend(loc="best")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("calibration_curve_test_manual.png")
plt.close()
print("Saved: calibration_curve_test_manual.png")

# %%
# --- APPEND: Confusion matrix (test split) at operating threshold ---
y_test_pred_cal = (y_test_prob_cal >= operating_threshold).astype(int)
cm = confusion_matrix(y_test, y_test_pred_cal)

fig, ax = plt.subplots(figsize=(6, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
ax.set_title(f"Confusion Matrix (test, threshold={operating_threshold:.2f})")
plt.tight_layout()
plt.savefig("confusion_matrix_test_manual.png")
plt.close(fig)
print("Saved: confusion_matrix_test_manual.png")

# Also print thresholded precision/recall for quick reference
prec = precision_score(y_test, y_test_pred_cal, zero_division=0)
rec  = recall_score(y_test, y_test_pred_cal)
print(f"Test precision @t={operating_threshold:.2f}: {prec:.4f} | recall: {rec:.4f}")

# %%
# --- APPEND: PR vs C (via Average Precision from cv_results_) ---
cv = pd.DataFrame(grid.cv_results_).copy()

# Extract numeric C
c_col = "param_classifier__C" if "param_classifier__C" in cv.columns else "param_C"
cv["C"] = cv[c_col].astype(float)

# Identify solver/penalty columns (if present)
solver_col  = "param_classifier__solver"  if "param_classifier__solver"  in cv.columns else ("param_solver"  if "param_solver"  in cv.columns else None)
penalty_col = "param_classifier__penalty" if "param_classifier__penalty" in cv.columns else ("param_penalty" if "param_penalty" in cv.columns else None)

# Plot AP vs C by solver/penalty
plt.figure(figsize=(9, 6))
group_cols = [c for c in [solver_col, penalty_col] if c is not None]
if group_cols:
    for name, g in cv.groupby(group_cols):
        # stable sorting on C
        g = g.sort_values("C")
        label = " / ".join(map(str, name)) if isinstance(name, tuple) else str(name)
        plt.plot(g["C"], g["mean_test_score"], marker="o", lw=2, label=label)
else:
    g = cv.sort_values("C")
    plt.plot(g["C"], g["mean_test_score"], marker="o", lw=2, label="Average Precision")

plt.xscale("log")
plt.xlabel("C (log scale)")
plt.ylabel("Mean test score (Average Precision)")
plt.title("Average Precision vs C (from GridSearchCV cv_results_)")
plt.grid(True, which="both", linestyle="--", alpha=0.4)
plt.legend(title="solver / penalty" if group_cols else None)
plt.tight_layout()
plt.savefig("ap_vs_c_by_solver_penalty.png")
plt.close()
print("Saved: ap_vs_c_by_solver_penalty.png")

# Optional aggregate view: best AP per C across all other params
best_per_c = cv.groupby("C")["mean_test_score"].max().reset_index().sort_values("C")
plt.figure(figsize=(8, 5))
plt.plot(best_per_c["C"], best_per_c["mean_test_score"], marker="o", lw=2, color="purple")
plt.xscale("log")
plt.xlabel("C (log scale)")
plt.ylabel("Best Mean AP across settings")
plt.title("Best Average Precision vs C (aggregate)")
plt.grid(True, which="both", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("ap_vs_c_best_aggregate.png")
plt.close()
print("Saved: ap_vs_c_best_aggregate.png")

# If multi-metric scoring was used and precision/recall means were recorded, plot them too
prec_key = "mean_test_precision"
rec_key  = "mean_test_recall"
if prec_key in cv.columns and rec_key in cv.columns:
    plt.figure(figsize=(9, 6))
    for name, g in cv.groupby(group_cols) if group_cols else [(None, cv)]:
        g = g.sort_values("C")
        label = " / ".join(map(str, name)) if isinstance(name, tuple) else (str(name) if name is not None else "All")
        plt.plot(g["C"], g[prec_key], marker="o", lw=2, label=f"Precision: {label}")
        plt.plot(g["C"], g[rec_key],  marker="s", lw=2, label=f"Recall: {label}")
    plt.xscale("log")
    plt.xlabel("C (log scale)")
    plt.ylabel("Mean test metric")
    plt.title("Precision and Recall vs C (if available in cv_results_)")
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig("precision_recall_vs_c.png")
    plt.close()
    print("Saved: precision_recall_vs_c.png")
else:
    print("Note: Individual precision/recall per C are unavailable because GridSearchCV scoring was set to 'average_precision' only; plotted Average Precision vs C instead.")

# %%
# --- APPEND: PR curve (test split) with operating point ---
prec, rec, thr = precision_recall_curve(y_test, y_test_prob_cal)

plt.figure(figsize=(8, 6))
plt.plot(rec, prec, lw=2, label="Calibrated PR curve")
# Mark the selected operating point
y_test_pred_cal = (y_test_prob_cal >= operating_threshold).astype(int)
p_op = precision_score(y_test, y_test_pred_cal, zero_division=0)
r_op = recall_score(y_test, y_test_pred_cal)
plt.scatter([r_op], [p_op], color="red", zorder=5, label=f"Operating t={operating_threshold:.2f}\nP={p_op:.2f}, R={r_op:.2f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curve (test split)")
plt.grid(alpha=0.3)
plt.legend(loc="best")
plt.tight_layout()
plt.savefig("precision_recall_curve_test_manual.png")
plt.close()
print("Saved: precision_recall_curve_test_manual.png")

# %%
