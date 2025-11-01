# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import traceback
import emoji
import pandas as pd
import os
import random
import numpy as np # Added for helper functions

# Ensure the training-time helper is importable for the pickled pipeline
# The pickled TfidfVectorizer(preprocessor=helpers.emoji_to_text) needs this symbol at load/use time
try:
    import helpers as _helpers  # module must exist as it did during training
    from helpers import (
        emoji_to_text as _unused_emoji_to_text,
    )  # ensure symbol is resolvable
except Exception:
    _helpers = None
    _unused_emoji_to_text = None


# -----------------------------
# Utilities (used by quiz only)
# -----------------------------
def emoji_to_text(text):
    return emoji.demojize(text or "")


def normalize_label(v):
    if v is None:
        return None
    if isinstance(v, str):
        s = v.strip().lower().strip("\"'").rstrip(",.;:")
        if s == "cg":
            return "Fake"
        if s == "or":
            return "Genuine"
        return None
    try:
        return "Fake" if int(v) == 1 else "Genuine"
    except Exception:
        return None


def stratified_sample_by_answer(df, n):
    """
    Draw a balanced sample across 'answer' classes when both classes exist.
    Falls back gracefully if a class is too small or only one class is present.
    """
    if df.empty:
        return df

    groups = list(df.groupby("answer"))
    if len(groups) <= 1:
        return df.sample(n=min(n, len(df)), replace=False, random_state=None)

    per = max(1, n // len(groups))
    parts = []
    for _, g in groups:
        k = min(per, len(g))
        parts.append(g.sample(n=k, replace=False, random_state=None))
    out = pd.concat(parts, axis=0)

    if len(out) < n:
        remaining = df.drop(out.index)
        if not remaining.empty:
            k = min(n - len(out), len(remaining))
            out = pd.concat(
                [out, remaining.sample(n=k, replace=False, random_state=None)], axis=0
            )

    return out.sample(frac=1.0, replace=False, random_state=None).head(n)


# -----------------------------
# App init
# -----------------------------
app = Flask(__name__)
CORS(app)

# -----------------------------
# Load model
# -----------------------------
print("Loading the model...")
model = None
threshold = 0.5
try:
    artifact = joblib.load(
        "model_logreg_calibrated.pkl"
    )  # {'model': CalibratedClassifierCV(Pipeline(...)), 'threshold': float, ...}
    model = artifact["model"]
    threshold = float(artifact.get("threshold", 0.5))
    print("Model loaded successfully.")
except FileNotFoundError:
    print(
        "Error: 'model_logreg_calibrated.pkl' not found. The API will not work without the model file."
    )
except Exception as e:
    print(f"Error loading model: {e}")

# --- NEW: EXPLAINER INITIALIZATION ---
# We extract the components from the pipeline to build explanations
explainer_components = {}
if model:
    try:
        # 1. Get the fitted pipeline from the calibrator
        pipeline = model.estimator
        
        # 2. Get the preprocessor (ColumnTransformer)
        preprocessor = pipeline.named_steps['preprocessor']
        
        # 3. Get the final logistic regression classifier
        classifier = pipeline.named_steps['classifier']
        
        # 4. Get the coefficients (weights) for the "Fake" class (index [0])
        #    (Class 0 is 'Genuine', Class 1 is 'Fake')
        coefficients = classifier.coef_[0]
        
        # 5. Get all feature names in the correct order from the preprocessor
        feature_names = preprocessor.get_feature_names_out()
        
        # Store them for later use
        explainer_components = {
            "preprocessor": preprocessor,
            "coefficients": coefficients,
            "feature_names": feature_names,
            "model": model, # Store the full model for prediction
        }
        print("Explainer initialized successfully.")
    except Exception as e:
        print(f"Could not initialize explainer: {e}")

# --- NEW: EXPLAINER HELPER FUNCTIONS ---

def _format_feature_name(name):
    """Cleans up the raw scikit-learn feature name."""
    if name.startswith('tfidf__'):
        return f"Word: '{name[7:]}'"
    # This check works even if the feature name is 'repetition_feature__x0'
    if name.startswith('repetition_feature__'):
        return "Repetition Score"
    if name.startswith('numeric__'):
        return "Star Rating"
    return name

def get_explanation_data(input_df, top_n=5):
    """
    Generates a feature-by-feature explanation for a single prediction.
    'input_df' must be a 1-row DataFrame matching the model's input.
    """
    if not explainer_components:
        return {"error": "Explainer is not initialized."}
    
    try:
        # Get components
        preprocessor = explainer_components["preprocessor"]
        coefficients = explainer_components["coefficients"]
        feature_names = explainer_components["feature_names"]

        # 1. Transform the raw input (text, rating) into a feature vector
        transformed_vector = preprocessor.transform(input_df)
        
        # 2. Get the dense feature values (e.g., TF-IDF scores, ratio, rating)
        feature_values = transformed_vector.toarray()[0]
        
        # 3. Calculate contribution: (Contribution = Value * Weight)
        contributions = feature_values * coefficients
        
        # 4. Map names to contributions
        contrib_list = list(zip(feature_names, contributions))
        
        # 5. Filter out features that had zero contribution (e.g., words not in review)
        contrib_list = [c for c in contrib_list if c[1] != 0]
        
        # 6. Sort to find top drivers
        contrib_list.sort(key=lambda x: x[1], reverse=True)
        
        # 7. Format the output
        top_positive = contrib_list[:top_n]
        top_negative = sorted(contrib_list, key=lambda x: x[1])[:top_n]

        return {
            "top_fake_drivers": [
                {"feature": _format_feature_name(name), "contribution": round(val, 3)}
                for name, val in top_positive if val > 0
            ],
            "top_genuine_drivers": [
                {"feature": _format_feature_name(name), "contribution": round(val, 3)}
                for name, val in top_negative if val < 0
            ]
        }
    
    except Exception as e:
        print(f"Error during explanation: {e}")
        traceback.print_exc()
        return {"error": str(e)}

def generate_explanation_text(explanation_data, genuine_label):
    """Generates a simple, human-readable text explanation."""
    
    if "error" in explanation_data:
        return "Could not generate an explanation."

    try:
        top_fake = explanation_data.get("top_fake_drivers", [])
        top_genuine = explanation_data.get("top_genuine_drivers", [])

        if genuine_label == "Fake":
            # Explain why it's FAKE
            if not top_fake:
                return f"The model rated this as **Fake**, but could not identify a strong reason."
            
            top_driver = top_fake[0]['feature']
            text = f"The model rated this as **Fake**. The strongest 'Fake' factor was the **{top_driver}**."
            
            if top_genuine:
                neg_driver = top_genuine[0]['feature']
                text += f" This outweighed 'Genuine' factors like the **{neg_driver}**."
            return text
        
        else:
            # Explain why it's GENUINE
            if not top_genuine:
                return f"The model rated this as **Genuine**, but could not identify a strong reason."
            
            top_driver = top_genuine[0]['feature']
            text = f"The model rated this as **Genuine**. The strongest 'Genuine' factor was the **{top_driver}**."
            
            if top_fake:
                pos_driver = top_fake[0]['feature']
                text += f" This outweighed 'Fake' factors like the **{pos_driver}**."
            return text
            
    except Exception as e:
        print(f"Error generating text explanation: {e}")
        return "An error occurred while summarizing the explanation."

# --- END NEW EXPLAINER FUNCTIONS ---


# -----------------------------
# Load training data (CSV schema: category,rating,label,text)
# -----------------------------
TRAIN_DATA_PATH = os.getenv("TRAIN_DATA_PATH", "./data/dataset.csv")

try:
    raw_df = pd.read_csv(TRAIN_DATA_PATH)

    # --- MODIFIED: Ensure 'rating', 'text', and 'label' are present ---
    required_cols = {"text", "label", "rating"}
    if not required_cols.issubset(raw_df.columns):
        print("CSV must contain 'text', 'label', and 'rating' columns for the quiz.")
        train_df = None
    else:
        # Keep 'rating' so the explainer can use it
        train_df = raw_df[["text", "label", "rating"]].dropna()
    # --- END MODIFICATION ---
        
        train_df["text"] = train_df["text"].astype(str).map(emoji_to_text).str.strip()
        train_df = train_df[train_df["text"].str.len() > 0]
        train_df["answer"] = train_df["label"].map(normalize_label)
        train_df = train_df.dropna(subset=["answer"])
        train_df["text"] = train_df["text"].str.slice(0, 500)
        train_df.reset_index(drop=True, inplace=True)
        print(
            "Quiz class distribution:",
            train_df["answer"].value_counts(dropna=False).to_dict(),
        )
        print(f"Quiz training pool size: {len(train_df)}")
except Exception as e:
    print(f"Failed to load training data for quiz: {e}")
    train_df = None


# -----------------------------
# Routes
# -----------------------------
@app.route("/predict", methods=["POST"], strict_slashes=False)
def predict():
    """
    Expects JSON: {"reviews": [{"text": "...", "rating": 4.5, "category": "..."} , ...]}.
    Returns: {"scores": [p1, p2, ...]} where each p is probability of "Fake".
    """
    if model is None:
        return jsonify({"error": "Model is not loaded, check server logs."}), 500

    try:
        data = request.get_json(force=True) or {}
        reviews = data.get("reviews", [])
        if not reviews or not isinstance(reviews, list):
            return jsonify({"error": "'reviews' must be a non-empty list."}), 400

        df = pd.DataFrame(reviews)

        # Validate required columns expected by the ColumnTransformer: text and rating
        required = {"text", "rating"}
        missing = [c for c in required if c not in df.columns]
        if missing:
            return jsonify({"error": f"Missing required fields: {missing}"}), 400

        # Coerce types to match training expectations
        df = df.copy()
        df["text"] = df["text"].astype(str)
        df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

        # Reject rows with NaN rating to avoid estimator errors
        if df["rating"].isna().any():
            return jsonify({"error": "All 'rating' values must be numeric."}), 400

        # Predict calibrated probabilities
        preds = model.predict_proba(df)
        scores = preds[:, 1].tolist()  # probability of the positive/"fake" class

        # Keep response identical to previous frontend expectations
        return jsonify({"scores": scores})
    except Exception as e:
        print("An error occurred during prediction:")
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 400


@app.route("/quiz-questions", methods=["GET"], strict_slashes=False)
def quiz_questions():
    """
    Returns n random questions with choices ["Genuine", "Fake"].
    Params:
      - n: number of questions (default 10)
      - stratify: 'true'/'false' (default 'true') for balanced sampling when both classes exist
      - require_mix: 'true'/'false' (default 'true'); if true and only one class exists, returns 422 with distribution
    """
    if train_df is None or train_df.empty:
        return jsonify({"error": "Training data not available for quiz."}), 500

    try:
        n = request.args.get("n", default=10, type=int)
        n = max(1, min(n, len(train_df)))
        stratify = (
            request.args.get("stratify", default="true").strip().lower() == "true"
        )
        require_mix = (
            request.args.get("require_mix", default="true").strip().lower() == "true"
        )

        dist = train_df["answer"].value_counts().to_dict()
        has_genuine = "Genuine" in dist and dist["Genuine"] > 0
        has_fake = "Fake" in dist and dist["Fake"] > 0

        if require_mix and not (has_genuine and has_fake):
            return (
                jsonify(
                    {
                        "error": "Only one class present in training pool; cannot produce a mixed quiz.",
                        "distribution": dist,
                        "single_class": True,
                    }
                ),
                422,
            )

        if stratify and has_genuine and has_fake:
            sample = stratified_sample_by_answer(train_df, n)
        else:
            sample = train_df.sample(n=n, replace=False, random_state=None)

        questions = []
        for i, row in enumerate(sample.itertuples(index=False), start=1):
            choices = ["Genuine", "Fake"]
            random.shuffle(choices)
            correct_label = row.answer # This is "Genuine" or "Fake"
            answer_index = choices.index(correct_label)
            
            # --- NEW: GENERATE EXPLANATION ---
            # Create the 1-row DataFrame that the model expects
            explainer_input_df = pd.DataFrame([
                {"text": row.text, "rating": row.rating}
            ])
            
            # Get the raw explanation data
            explanation_data = get_explanation_data(explainer_input_df)
            
            # Get the simple text explanation
            explanation_text = generate_explanation_text(explanation_data, correct_label)
            # --- END NEW BLOCK ---
            
            questions.append(
                {
                    "qid": f"q{i}",
                    "prompt": row.text,
                    "choices": choices,
                    "answer_index": answer_index,
                    "explanation_data": explanation_data, # NEW: For detailed UI
                    "explanation_text": explanation_text  # NEW: For simple text display
                }
            )

        return jsonify(
            {
                "questions": questions,
                "count": len(questions),
                "distribution": dist,
                "single_class": not (has_genuine and has_fake),
            }
        )
    except Exception as e:
        print("An error occurred building quiz questions:")
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 400


# -----------------------------
# Entrypoint
# -----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5051, debug=True)
