# helpers.py
import re
import numpy as np
import emoji
from sklearn.base import BaseEstimator, TransformerMixin


def emoji_to_text(text):
    """Converts emojis in a string to their text representation."""
    return emoji.demojize(text)


class RepetitionFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Creates a feature that is the ratio of unique words to total words.
    A low ratio indicates high repetition; a high ratio indicates no repetition.
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def _get_unique_word_ratio(self, text):
        words = re.findall(r"\b\w+\b", (text or "").lower())
        if not words:
            return 0.5
        total_words = len(words)
        unique_words = len(set(words))
        return unique_words / total_words

    def transform(self, X_series, y=None):
        return X_series.apply(self._get_unique_word_ratio).values.reshape(-1, 1)

    def get_feature_names_out(self, input_features=None):
        # ColumnTransformer will prefix the transformer name, yielding "repetition_feature__x0"
        return np.array(["x0"])
