import nltk

from typing import List, Text
from tqdm import tqdm

from nltk import tree
from nltk import WordNetLemmatizer
from nltk import PorterStemmer
from nltk.sentiment import SentimentIntensityAnalyzer

from app.preprocessing.text_normalizer import TextNormalizer


class TextPreparation:

    def __init__(
        self,
        text: str,
    ):
        self.text_normalizer: TextNormalizer = TextNormalizer(text)

    @property
    def text_removed_from_rare_words(self) -> Text:
        """
        Retrieve cleared text including stop words.
        :return:
            Text - text with  stop words.
        """
        self.cleaned_text(remove_rare_words=True)
        return self.text_normalizer.text

    def cleaned_text(self, del_stopwords=False, remove_rare_words=False) -> List[Text]:
        """
        Retrieve cleared text.
        :arg:
            del_stopwords (Optional): - remove stop words from a text corpus.
            remove_rare_words (Optional): remove rare words from a text corpus.
        :return:
            Text - cleared text.
        """
        return self.text_normalizer.normalize(del_stopwords=del_stopwords, remove_rare_words=remove_rare_words)

    def stemmed(self) -> List[str]:
        """
        Stemming is the process of reducing the tokens to their root forms, which are not necessarily valid words.
        For example, the words 'running', 'runs', and 'run' can be stemmed to the root form 'run'.
        Stemming can help group together words that have similar meanings but different forms.
        :return:
            list - stemmed words.
        """

        stemmer: PorterStemmer = PorterStemmer()
        filtered_tokens = self.cleaned_text(del_stopwords=True)
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
        :return:
            list - lemmatized words.
        """
        lemmatizer: WordNetLemmatizer = WordNetLemmatizer()
        filtered_tokens = self.cleaned_text(del_stopwords=True)
        lemmatized_words = set()
        for word in tqdm(filtered_tokens):
            lemmatized_word = lemmatizer.lemmatize(word)
            lemmatized_words.add(lemmatized_word)
        lemmatized_tokens = list(lemmatized_words)
        return lemmatized_tokens

    def extract_parts_of_speech(self) -> tree.Tree:
        """
        Part-of-speech tagging is the process of assigning a grammatical category to each
        token, such as noun, verb, adjective, etc.
        Part-of-speech tagging can help us understand the structure and meaning of the text.
        :return:
            tree.Tree - NLTK chunks the tagged tokens into named entities based on the IOB notation, which stands
            for Inside-Outside-Beginning.
            This notation indicates whether a token is inside a named entity (I), outside a named entity (O), or at the
            beginning of a named entity (B).
            You can find more information about the IOB notation clicking
            the link https://en.wikipedia.org/wiki/Inside%E2%80%93outside%E2%80%93beginning_(tagging).
        """
        pos_tag_tokens = nltk.word_tokenize(self.text_removed_from_rare_words)
        tags = nltk.pos_tag(pos_tag_tokens)
        chunks = nltk.ne_chunk(tags)
        return chunks

    def sentiment_analysis(self) -> dict:
        """
        Give a sentiment intensity score to sentences.
        :return:
            Dict - return a float for sentiment strength based on the input text.
            Positive values are positive valence, negative value are negative valence.
            Note:
                Hashtags are not taken into consideration (e.g. #BAD is neutral).
                If you are interested in processing the text in the hashtags too, then
                we recommend preprocessing your data to remove the #, after
                which the hashtag text may be matched as if it was a normal word in the sentence.
        """
        sia = SentimentIntensityAnalyzer()
        scores = sia.polarity_scores(self.text_removed_from_rare_words)
        return scores
