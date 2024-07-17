import re
import nltk

from collections import Counter

from app.preprocessing.tools import get_stopwords


class TextNormalizer:

    def __init__(self, text):
        self.text = text
        self.to_lowercase()
        self.delete_punctuation()
        self.delete_numbers()

        self.words = self.tokenize(self.text)

    @staticmethod
    def tokenize(text):
        words = nltk.word_tokenize(text)
        return words

    def to_lowercase(self):
        self.text = self.text.lower()

    def delete_punctuation(self):
        self.text = re.sub(r"[^\w\s]", "", self.text)

    def delete_numbers(self):
        self.text = re.sub(r"\d", "", self.text)

    def remove_rare_words(self, threshold=2):
        words = self.tokenize(self.text)
        word_freq = Counter(words)
        filtered_words = [word for word in self.words if word_freq[word] >= threshold]
        self.text = " ".join(filtered_words)

    def delete_stopwords(self, stop_words):
        self.words = [token for token in self.words if token not in stop_words]

    def normalize(self, del_stopwords=False, remove_rare_words=False):
        if remove_rare_words:
            self.remove_rare_words()
        if del_stopwords:
            self.delete_stopwords(stop_words=get_stopwords())
        return self.words
