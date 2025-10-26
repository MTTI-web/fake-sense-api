# create_dummy_model.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib

print("Creating a dummy model...")

# Sample data: just enough to make the model work
data = {
    "text": [
        "This is an amazing product, I love it!",
        "Worst purchase ever, completely broken.",
        "Best buy ever!! great quality and price",
        "I demand a refund this is a scam",
    ],
    "label": [0, 1, 0, 1],  # 0 for "real", 1 for "fake"
}
df = pd.read_csv("./data/dataset.csv")

# Create a scikit-learn pipeline
# This pipeline first converts text to numbers (TF-IDF), then classifies it.
model_pipeline = Pipeline(
    [("tfidf", TfidfVectorizer()), ("classifier", LogisticRegression())]
)

# "Train" the model on our tiny dataset
model_pipeline.fit(df["text"], df["label"])

# Save the trained model to a file
joblib.dump(model_pipeline, "fake_review_model.pkl")

print("Dummy model 'fake_review_model.pkl' created successfully!")
