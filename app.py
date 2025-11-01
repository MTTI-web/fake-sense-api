# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import traceback
import emoji
import pandas as pd
import os
import random

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

# -----------------------------
# Load training data (CSV schema: category,rating,label,text)
# -----------------------------
TRAIN_DATA_PATH = os.getenv("TRAIN_DATA_PATH", "./data/dataset.csv")

try:
    raw_df = pd.read_csv(TRAIN_DATA_PATH)

    if not {"text", "label"}.issubset(raw_df.columns):
        print("CSV must contain 'text' and 'label' columns.")
        train_df = None
    else:
        train_df = raw_df[["text", "label"]].dropna()
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
            correct = row.answer
            answer_index = choices.index(correct)
            questions.append(
                {
                    "qid": f"q{i}",
                    "prompt": row.text,
                    "choices": choices,
                    "answer_index": answer_index,
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
