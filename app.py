# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS  # Import the CORS library
import joblib
import traceback  # For detailed error logging

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
    model = joblib.load("fake_review_model.pkl")
    print("Model loaded successfully.")
except FileNotFoundError:
    print(
        "Error: 'fake_review_model.pkl' not found. The API will not work without the model file."
    )
    model = None
except Exception as e:
    print(f"Error loading model: {e}")
    model = None


# Define the API endpoint for making predictions
@app.route("/predict", methods=["POST"])
def predict():
    if not model:
        return jsonify({"error": "Model is not loaded, check server logs."}), 500

    try:
        # Get the JSON data sent from the browser extension
        data = request.get_json(force=True)

        # The data should be in a format like: {"reviews": ["text1", "text2", ...]}
        review_texts = data["reviews"]

        # Use the model to predict the probability of being "fake"
        # predict_proba returns probabilities for each class: [P(real), P(fake)]
        predictions = model.predict_proba(review_texts)

        # We only need the probability of the "fake" class (the second column)
        fakery_scores = predictions[:, 1].tolist()

        # Return the scores as a JSON response
        return jsonify({"scores": fakery_scores})

    except Exception as e:
        # If anything goes wrong, log the error and return an error message
        print("An error occurred during prediction:")
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 400


# This allows you to run the app directly using "python app.py"
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5051, debug=True)
