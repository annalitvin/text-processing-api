import joblib
import nltk

from os.path import join

from app.preprocessing.config import config
from app.preprocessing.text_normalizer import TextNormalizer
from app.preprocessing.text_preparation import TextPreparation
from app.preprocessing.tools import upload_stopwords


def text_preprocessing(
    raw_text,
    cleaned_text=False,
    stemmed_text=False,
    lemmatize_text=False,
    parts_of_speech=False,
    sentiment_analysis=False,
):
    # Download NLTK data
    nltk.download("punkt")
    nltk.download("averaged_perceptron_tagger")
    nltk.download("maxent_ne_chunker")
    nltk.download("words")
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)
    nltk.download("vader_lexicon", quiet=True)
    # Download stopwords
    upload_stopwords()

    preprocessing_result = dict()

    text_preparation = TextPreparation(raw_text)
    if cleaned_text:
        text_normalizer = TextNormalizer(raw_text)
        cleaned_text = text_normalizer.normalize(del_stopwords=True, remove_rare_words=True)
        preprocessing_result["cleaned_text"] = cleaned_text
    if stemmed_text:
        stemmed_tokens = text_preparation.stemmed()
        preprocessing_result["stemmed_tokens"] = stemmed_tokens
    if lemmatize_text:
        lemmatized_tokens = text_preparation.lemmatize()
        preprocessing_result["lemmatized_tokens"] = lemmatized_tokens
    if parts_of_speech:
        parts_of_speech_tags = text_preparation.extract_parts_of_speech()
        preprocessing_result["parts_of_speech_tags"] = parts_of_speech_tags
    if sentiment_analysis:
        scores = text_preparation.sentiment_analysis()
        preprocessing_result["sentiment_analysis_scores"] = scores

    joblib.dump(preprocessing_result, join(config.preprocessing_result_file_path, "preprocessing_result.pkl"))
    return preprocessing_result


if __name__ == "__main__":
    raw_text_file_path = join(config.raw_text_file_path, "raw_text.txt")
    test_text = open(raw_text_file_path, "r").read()
    result = text_preprocessing(test_text, True, True, True, True, True)
