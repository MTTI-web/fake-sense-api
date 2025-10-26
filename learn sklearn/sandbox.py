# %% seeing what tfidfvectorizer does

from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

data = {
    "text": [
        "This is an amazing product, I love it!",
        "Worst purchase ever, completely broken.",
        "Best buy ever!! great quality and price",
        "I demand a refund this is a scam",
    ],
    "label": [0, 1, 0, 1],  # 0 for "real", 1 for "fake"
}

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data)
# Convert to array
array = X.toarray()
# Get feature names (words)
feature_names = vectorizer.get_feature_names_out()
# Create DataFrame
df = pd.DataFrame(array, columns=feature_names)
print(df)

# %%
