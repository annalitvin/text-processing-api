import re
import nltk

from collections import Counter
from typing import Text

from app.preprocessing.tools import get_stopwords


class TextNormalizer:

    def __init__(self, raw_text: Text):
        self.text: Text = raw_text
        self.to_lowercase()
        self.delete_punctuation()
        self.delete_numbers()

        self.words: list = self.tokenize(self.text)

    @staticmethod
    def tokenize(text) -> list:
        words = nltk.word_tokenize(text)
        return words

    def to_lowercase(self):
        """Convert all characters to lowercase from list of tokenized words"""
        self.text = self.text.lower()

    def delete_punctuation(self):
        """Delete punctuation from text"""
        self.text = re.sub(r"[^\w\s]", "", self.text)

    def delete_numbers(self):
        """Delete numbers from list of tokenized words"""
        self.text = re.sub(r"\d", "", self.text)

    def remove_rare_words(self, threshold: int = 2):
        """
        Remove rare words from list of tokenized words
        :arg:
            threshold: int - number of times a word appears.
        """
        word_freq = Counter(self.words)
        filtered_words = [word for word in self.words if word_freq[word] >= threshold]
        self.text = " ".join(filtered_words)
        self.words = filtered_words

    def delete_stopwords(self, stop_words: list):
        """Remove stop words from list of tokenized words
        :arg:
            stop_words: list - Stop words are the words in a stop list (or stoplist or negative dictionary)
            which are filtered out (i.e. stopped) before or after processing of natural language data (text)
            because they are deemed insignificant.
        """
        self.words = [token for token in self.words if token not in stop_words]

    def normalize(self, del_stopwords: bool = False, remove_rare_words: bool = False):
        if remove_rare_words:
            self.remove_rare_words()
        if del_stopwords:
            self.delete_stopwords(stop_words=get_stopwords())
        return self.words
