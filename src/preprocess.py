import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

nltk.download("stopwords")
stop_words = set(stopwords.words("english"))
stemmer = SnowballStemmer("english")

def preprocess_text(text):
    """
    Nettoyage du texte :
    - minuscules
    - enlever ponctuation
    - enlever stopwords
    - stemming
    """
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    words = text.split()
    words = [stemmer.stem(w) for w in words if w not in stop_words]
    return ' '.join(words)
