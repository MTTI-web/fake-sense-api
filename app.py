# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS  # Import the CORS library
import joblib
import traceback  # For detailed error logging
import emoji


def emoji_to_text(text):
    return emoji.demojize(text)


# Initialize the Flask application
app = Flask(__name__)

# --- CORS Configuration ---
# This is the key change. We enable CORS for the entire app,
# allowing your browser extension (from any origin) to make requests.
CORS(app)
# -------------------------

# Load the trained machine learning model from the file
print("Loading the model...")
try:
    # Ensure you have a 'fake_review_model.pkl' file in the same directory
    model = joblib.load("model.pkl")
    print("Model loaded successfully.")
except FileNotFoundError:
    print(
        "Error: 'fake_review_model.pkl' not found. The API will not work without the model file."
    )
    model = None
except Exception as e:
    print(f"Error loading model: {e}")
    model = None


from flask import Flask, request, jsonify
import traceback
import pandas as pd

# ... model loading code ...


@app.route("/predict", methods=["POST"])
def predict():
    if not model:
        return jsonify({"error": "Model is not loaded, check server logs."}), 500

    try:
        # Get the JSON data sent from the browser extension
        data = request.get_json(force=True)

        # The data should be in the format:
        # {"reviews": [{"text": ..., "rating": ...}, ...]}
        reviews = data.get("reviews", [])
        print(reviews)

        if not reviews or not isinstance(reviews, list):
            return jsonify({"error": "'reviews' must be a non-empty list."}), 400

        # Build a DataFrame with columns matching model's training
        df = pd.DataFrame(reviews)

        # Use the model to predict probabilities
        predictions = model.predict_proba(df)

        # Extract the probability of the "fake" class (second column)
        fakery_scores = predictions[:, 1].tolist()

        # Return the scores as a JSON response
        return jsonify({"scores": fakery_scores})

    except Exception as e:
        print("An error occurred during prediction:")
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 400


# This allows you to run the app directly using "python app.py"
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5051, debug=True)
