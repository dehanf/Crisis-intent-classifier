import re
import string
from nltk.stem import PorterStemmer

# Initialize the stemmer
stemmer = PorterStemmer()

def clean_text(text):
    """
    Cleans raw text: lowercases, removes punctuation, and stems words.
    Example: '5 people are trapped!' -> '5 peopl are trap'
    """
    if not isinstance(text, str):
        return ""

    # 1. Lowercase
    text = text.lower()
    
    # 2. Remove special characters and punctuation
    # This keeps alphanumeric characters but removes things like @, #, !, etc.
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    
    # 3. Stemming
    # Split sentence into words, stem each word, and join back
    words = text.split()
    stemmed_words = [stemmer.stem(word) for word in words]
    
    return " ".join(stemmed_words)