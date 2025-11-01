# %% [markdown]
# # Model Analysis Notebook (Full)
#
# This script re-trains the model from `create_model.py`
# to generate diagnostic plots for your presentation.
#
# It will produce 4 graph files:
# 1. `c_value_vs_precision.png`
# 2. `calibration_curve.png`
# 3. `roc_curve.png`
# 4. `feature_importance.png`

# %%
# --- Imports ---
import numpy as np
import pandas as pd
import joblib
import re
from sklearn.base import BaseEstimator, TransformerMixin

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay
from sklearn.metrics import precision_recall_curve, RocCurveDisplay

# Must have 'helpers.py' in the same folder
from helpers import emoji_to_text

# %%
# --- Custom Transformer (Copied from create_model.py) ---
# We must define the class again so joblib can use it
class RepetitionFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, X, y=None):
        return self
    def _get_unique_word_ratio(self, text):
        words = re.findall(r'\b\w+\b', (text or "").lower())
        if not words:
            return 0.5
        total_words = len(words)
        unique_words = len(set(words))
        return unique_words / total_words
    def transform(self, X_series, y=None):
        return X_series.apply(self._get_unique_word_ratio).values.reshape(-1, 1)

# %%
# --- Load and Split Data (Same as create_model.py) ---
print("Loading and splitting data...")
try:
    df = pd.read_csv("./data/dataset.csv")
except FileNotFoundError:
    print("Error: './data/dataset.csv' not found.")
    print("Please run this script from the 'fake-sense-api' directory.")
    exit()
    
X_full = df[["rating", "text"]]
y_full = (df["label"] == "CG").astype(int)

X_train, X_temp, y_train, y_temp = train_test_split(
    X_full, y_full, test_size=0.4, stratify=y_full, random_state=42
)
X_calib, X_test, y_calib, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
)
print("Data ready.")

# %%
# --- Define Pipeline (Same as create_model.py) ---
preprocessor = ColumnTransformer(
    transformers=[
        ("tfidf", TfidfVectorizer(preprocessor=emoji_to_text), "text"),
        ("repetition_feature", RepetitionFeatureExtractor(), "text"),
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
# --- Run GridSearchCV (Same as create_model.py) ---
# We need to re-run this to get the results for all C values
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
    scoring="average_precision", # This is what we will plot
    cv=5,
    n_jobs=-1,
    refit=True,
)
print("Starting GridSearchCV... (This may take a minute)")
grid.fit(X_train, y_train)
best_lr_pipe = grid.best_estimator_
print(f"GridSearchCV complete. Best params: {grid.best_params_}")

# %%
# --- GRAPH 1: C-Value vs. Precision Score ---
# This graph shows how the model's score (Average Precision)
# changed with different C values and penalties.
print("Generating Graph 1: C-Value vs. Precision...")
try:
    results = pd.DataFrame(grid.cv_results_)
    
    # Filter for easier plotting
    liblinear_l1 = results[
        (results['param_classifier__solver'] == 'liblinear') & 
        (results['param_classifier__penalty'] == 'l1')
    ]
    liblinear_l2 = results[
        (results['param_classifier__solver'] == 'liblinear') & 
        (results['param_classifier__penalty'] == 'l2')
    ]
    
    plt.figure(figsize=(10, 7))
    plt.plot(
        liblinear_l1['param_classifier__C'], 
        liblinear_l1['mean_test_score'], 
        'bo-', 
        label='liblinear (L1 Penalty)'
    )
    plt.plot(
        liblinear_l2['param_classifier__C'], 
        liblinear_l2['mean_test_score'], 
        'rs-', 
        label='liblinear (L2 Penalty)'
    )
    
    plt.xscale('log') # C values are logarithmic
    plt.xlabel('C Value (Regularization Strength)')
    plt.ylabel('Mean Average Precision (Score)')
    plt.title('Model Performance vs. C Value')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    
    output_filename = 'c_value_vs_precision.png'
    plt.savefig(output_filename)
    print(f"Successfully saved graph to {output_filename}")
    # plt.show() # Use this if running interactively

except Exception as e:
    print(f"Could not generate C-Value graph: {e}")


# %%
# --- Calibrate Model (Same as create_model.py) ---
print("Calibrating model...")
calibrator = CalibratedClassifierCV(
    estimator=best_lr_pipe, method="sigmoid", cv="prefit"
)
calibrator.fit(X_calib, y_calib)
print("Calibration complete.")


# %%
# --- GRAPH 2: Calibration Curve ---
# This graph shows the "before" (uncalibrated) and "after" (calibrated)
# reliability of the model's probabilities.
print("Generating Graph 2: Calibration Curve...")
try:
    plt.figure(figsize=(10, 10))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))
    
    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly Calibrated")
    
    # Plot the "uncalibrated" model (best_lr_pipe)
    CalibrationDisplay.from_estimator(
        best_lr_pipe,
        X_calib,
        y_calib,
        n_bins=10,
        name="Uncalibrated (Before)",
        ax=ax1,
        strategy='uniform'
    )
    
    # Plot the "calibrated" model (calibrator)
    CalibrationDisplay.from_estimator(
        calibrator,
        X_calib,
        y_calib,
        n_bins=10,
        name="Calibrated (After)",
        ax=ax1,
        strategy='uniform'
    )
    
    # Show the distribution of probabilities
    ax2.hist(
        calibrator.predict_proba(X_calib)[:, 1],
        range=(0, 1),
        bins=10,
        label="Calibrated Probabilities",
        lw=2
    )
    ax2.set_xlabel("Predicted Probability (of 'Fake')")
    ax2.set_ylabel("Count")
    
    ax1.set_ylabel("Fraction of Positives (Actual)")
    ax1.set_title("Calibration Curve (Before vs. After)")
    ax1.legend()
    
    output_filename = 'calibration_curve.png'
    plt.tight_layout()
    plt.savefig(output_filename)
    print(f"Successfully saved graph to {output_filename}")
    # plt.show() # Use this if running interactively

except Exception as e:
    print(f"Could not generate Calibration graph: {e}")


# %%
# --- GRAPH 3: ROC Curve ---
# This graph shows the model's performance on the TEST set.
# This is a standard measure of a classifier's quality.
print("Generating Graph 3: ROC Curve...")
try:
    plt.figure(figsize=(8, 8))
    ax = plt.gca()
    
    # Plot the ROC curve for the final calibrated model on the TEST set
    RocCurveDisplay.from_estimator(
        calibrator,
        X_test,
        y_test,
        ax=ax,
        name="Calibrated Model (Test Set)"
    )
    
    # Plot the "random guess" line
    plt.plot([0, 1], [0, 1], 'r--', label='Random Guess')
    
    plt.title('ROC Curve for Final Model')
    plt.legend()
    
    output_filename = 'roc_curve.png'
    plt.savefig(output_filename)
    print(f"Successfully saved graph to {output_filename}")
    # plt.show() # Use this if running interactively

except Exception as e:
    print(f"Could not generate ROC graph: {e}")


# %%
# --- GRAPH 4: Feature Importance ---
# This graph shows WHAT the model learned.
# It plots the top 15 words/features that predict "Fake" (positive)
# and the top 15 that predict "Genuine" (negative).
print("Generating Graph 4: Feature Importance...")
try:
    # 1. Get the final logistic regression model
    final_logreg = best_lr_pipe.named_steps['classifier']
    
    # 2. Get the preprocessor
    final_preprocessor = best_lr_pipe.named_steps['preprocessor']
    
    # 3. Get all the feature names in the correct order
    tfidf_features = final_preprocessor.named_transformers_['tfidf'].get_feature_names_out()
    
    # Manually get the names in the same order the ColumnTransformer produced them
    feature_names = list(tfidf_features)
    feature_names.append('repetition_score') # This is the name from create_model.py
    feature_names.append('rating') # This is from the 'numeric' passthrough
    
    # 4. Get the coefficients (the "weights" or "importance")
    coefficients = final_logreg.coef_[0]
    
    # 5. Create a DataFrame to make this easy
    feat_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': coefficients
    }).sort_values(by='importance', ascending=False)
    
    # 6. Get the top 15 "Fake" predictors and top 15 "Genuine" predictors
    top_n = 15
    top_fake = feat_importance.head(top_n)
    top_genuine = feat_importance.tail(top_n).sort_values(by='importance', ascending=True)
    
    # Combine them for plotting
    plot_data = pd.concat([top_fake, top_genuine]).sort_values(by='importance')
    
    # 7. Create the bar chart
    plt.figure(figsize=(12, 12))
    
    # Assign colors: positive (fake) is red, negative (genuine) is green
    colors = ['red' if c > 0 else 'green' for c in plot_data['importance']]
    
    plt.barh(plot_data['feature'], plot_data['importance'], color=colors)
    plt.xlabel('Importance (Coefficient Weight)')
    plt.ylabel('Feature')
    plt.title(f'Top {top_n} "Fake" vs. "Genuine" Predictors')
    plt.tight_layout()
    
    output_filename = 'feature_importance.png'
    plt.savefig(output_filename)
    print(f"Successfully saved graph to {output_filename}")
    # plt.show() # Use this if running interactively

except Exception as e:
    print(f"Could not generate Feature Importance graph: {e}")

print("\nAnalysis complete. All graphs saved.")

# %%