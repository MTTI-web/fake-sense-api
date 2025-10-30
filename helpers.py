import emoji

def emoji_to_text(text):
    """Converts emojis in a string to their text representation."""
    return emoji.demojize(text)