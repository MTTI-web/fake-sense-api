# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import traceback
import emoji
import pandas as pd
import os
import random


# -----------------------------
# Utilities
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
        # Only one class available; return a regular sample
        return df.sample(
            n=min(n, len(df)), replace=False, random_state=None
        )  # pandas sampling [web:15]

    # Aim for an even split across classes
    per = max(1, n // len(groups))
    parts = []
    for _, g in groups:
        k = min(per, len(g))
        parts.append(
            g.sample(n=k, replace=False, random_state=None)
        )  # group-wise sample [web:116]

    out = pd.concat(parts, axis=0)

    # Fill remainder to reach n with leftover rows if needed
    if len(out) < n:
        remaining = df.drop(out.index)
        if not remaining.empty:
            k = min(n - len(out), len(remaining))
            out = pd.concat(
                [out, remaining.sample(n=k, replace=False, random_state=None)], axis=0
            )  # pandas sampling [web:15]

    # Shuffle and trim to n
    return out.sample(frac=1.0, replace=False, random_state=None).head(
        n
    )  # pandas sampling [web:15]


# -----------------------------
# App init
# -----------------------------
app = Flask(__name__)
CORS(app)  # enable during development [web:2]

# -----------------------------
# Load model
# -----------------------------
print("Loading the model...")
try:
    model = joblib.load("model.pkl")
    print("Model loaded successfully.")
except FileNotFoundError:
    print("Error: 'model.pkl' not found. The API will not work without the model file.")
    model = None
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# -----------------------------
# Load training data (CSV schema: category,rating,label,text)
# -----------------------------
TRAIN_DATA_PATH = os.getenv("TRAIN_DATA_PATH", "./data/dataset.csv")

try:
    raw_df = pd.read_csv(TRAIN_DATA_PATH)

    # Ensure required columns for quiz
    if not {"text", "label"}.issubset(raw_df.columns):
        print("CSV must contain 'text' and 'label' columns.")
        train_df = None
    else:
        # Keep only quiz-relevant columns and clean
        train_df = raw_df[["text", "label"]].dropna()
        train_df["text"] = train_df["text"].astype(str).map(emoji_to_text).str.strip()
        train_df = train_df[train_df["text"].str.len() > 0]

        # Normalize labels and drop unknowns so they don't bias to a single class
        train_df["answer"] = train_df["label"].map(normalize_label)
        train_df = train_df.dropna(subset=["answer"])

        # Optional: cap long texts for quiz readability
        train_df["text"] = train_df["text"].str.slice(0, 500)
        train_df.reset_index(drop=True, inplace=True)

        # Log class distribution
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
    The model should be a pipeline that knows how to handle the provided keys.
    """
    if not model:
        return (
            jsonify({"error": "Model is not loaded, check server logs."}),
            500,
        )  # Flask jsonify [web:2]

    try:
        data = request.get_json(force=True)
        reviews = data.get("reviews", [])
        if not reviews or not isinstance(reviews, list):
            return (
                jsonify({"error": "'reviews' must be a non-empty list."}),
                400,
            )  # Flask jsonify [web:2]

        df = pd.DataFrame(reviews)
        # The model pipeline should align columns internally (e.g., via ColumnTransformer)
        preds = model.predict_proba(df)
        scores = preds[:, 1].tolist()  # probability of the positive/"fake" class
        return jsonify({"scores": scores})  # Flask jsonify [web:2]
    except Exception as e:
        print("An error occurred during prediction:")
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 400  # Flask jsonify [web:2]


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
        return (
            jsonify({"error": "Training data not available for quiz."}),
            500,
        )  # Flask jsonify [web:2]

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
            )  # Flask jsonify [web:2]

        if stratify and has_genuine and has_fake:
            sample = stratified_sample_by_answer(
                train_df, n
            )  # group-wise sample [web:116]
        else:
            sample = train_df.sample(
                n=n, replace=False, random_state=None
            )  # pandas sampling [web:15]

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
        )  # Flask jsonify [web:2]
    except Exception as e:
        print("An error occurred building quiz questions:")
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 400  # Flask jsonify [web:2]


# -----------------------------
# Entrypoint
# -----------------------------
if __name__ == "__main__":
    # Accept both /quiz-questions and /quiz-questions/ thanks to strict_slashes=False on the route [web:2].
    app.run(
        host="0.0.0.0",
        port=5051,
        # debug=True
    )  # pandas sampling docs referenced above for selection logic [web:15]
