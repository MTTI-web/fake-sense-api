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
# ... (your code to calculate operating_threshold) ...
idx = np.where(precision[:-1] >= target_precision)[0]
operating_threshold = float(thresholds[idx[0]] if len(idx) else 0.5)

# --- ADD THIS ENTIRE BLOCK ---
try:
    print("\nGenerating Precision-Recall curve graph...")
    # Find the recall and precision for our chosen threshold
    # Note: np.where returns a tuple, so we access the first element
    chosen_point_idx = idx[0] if len(idx) > 0 else -1
    
    # Handle case where no threshold meets the target
    if chosen_point_idx == -1:
        print("Warning: No threshold met target precision. Marking best-effort point.")
        # Fallback: find point with highest precision
        chosen_point_idx = np.argmax(precision)

    chosen_recall = recall[chosen_point_idx]
    chosen_precision = precision[chosen_point_idx]
    
    # Create the plot
    plt.figure(figsize=(10, 7))
    plt.plot(recall, precision, 'b-', label='Precision-Recall curve')
    
    # Draw the target precision line
    plt.axhline(y=target_precision, color='r', linestyle='--', 
                label=f'Target Precision ({target_precision*100}%)')
    
    # Mark the chosen operating point
    plt.plot(chosen_recall, chosen_precision, 'ro', 
             label=f'Chosen Threshold ({operating_threshold:.2f})\n'
                   f'Precision: {chosen_precision:.2f}, Recall: {chosen_recall:.2f}')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve for Calibrated Model')
    plt.legend()
    plt.grid(True)
    
    # Save the figure
    output_filename = 'precision_recall_curve.png'
    plt.savefig(output_filename)
    print(f"Successfully saved graph to {output_filename}")
    
except Exception as e:
    print(f"Could not generate graph: {e}")
# --- END ADDED BLOCK ---

# %%
# Persist single artifact
artifact = {
    "model": calibrator,  # Calibrated, ready for predict_proba
    "threshold": operating_threshold,  # Operating point for classification
    "best_params": grid.best_params_,  # For reproducibility
}
joblib.dump(artifact, "model_logreg_calibrated.pkl")

# %%