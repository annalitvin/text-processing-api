import nltk

from typing import List
from tqdm import tqdm

from nltk import WordNetLemmatizer
from nltk import PorterStemmer
from nltk.sentiment import SentimentIntensityAnalyzer

from app.preprocessing.text_normalizer import TextNormalizer


class TextPreparation:

    def __init__(
        self,
        text: str,
    ):
        self.text_normalizer = TextNormalizer(text)

    @property
    def text_removed_from_rare_words(self):
        filtered_tokens = self.cleaned_text(del_stopwords=False, remove_rare_words=True)
        return " ".join(filtered_tokens)

    def cleaned_text(self, del_stopwords=False, remove_rare_words=False):
        return self.text_normalizer.normalize(del_stopwords=del_stopwords, remove_rare_words=remove_rare_words)

    def stemmed(self):
        stemmer: PorterStemmer = PorterStemmer()
        filtered_tokens = self.cleaned_text(del_stopwords=True, remove_rare_words=True)
        stemmed = set()
        for token in tqdm(filtered_tokens):
            stemmed_word = stemmer.stem(token)
            stemmed.add(stemmed_word)
        stemmed_tokens = list(stemmed)
        return stemmed_tokens

    def lemmatize(self) -> List[str]:
        """
        To reduce the different forms of a word to one single form,
        for example, reducing "builds", "building",or "built" to the lemma "build":
        Compounds were lemmatized, that is, inflectional differences were disregarded.
        Args:
            words: numpy array - words that are obtained from the dataset.
        returns:
            list - lemmatized words.
        """
        lemmatizer: WordNetLemmatizer = WordNetLemmatizer()
        filtered_tokens = self.cleaned_text(del_stopwords=True, remove_rare_words=True)
        lemmatized_words = set()
        for word in tqdm(filtered_tokens):
            lemmatized_word = lemmatizer.lemmatize(word)
            lemmatized_words.add(lemmatized_word)
        lemmatized_tokens = list(lemmatized_words)
        return lemmatized_tokens

    def extract_parts_of_speech(self):
        pos_tag_tokens = nltk.word_tokenize(self.text_removed_from_rare_words)
        tags = nltk.pos_tag(pos_tag_tokens)
        chunks = nltk.ne_chunk(tags)
        return chunks

    def sentiment_analysis(self):
        sia = SentimentIntensityAnalyzer()
        scores = sia.polarity_scores(self.text_removed_from_rare_words)
        return scores
