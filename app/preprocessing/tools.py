import pickle
from os.path import join
from nltk.corpus import stopwords as nltk_stopwords

from app.core.config import settings


def serialize(data, file_path):
    with open(file_path, "wb") as f:
        pickle.dump(data, f)


def upload_stopwords():
    """Load stopwords from all the sources we got"""
    extra_stopwords = set()
    stopwords_file = join(settings.RAW_STOPWORDS_FILE_PATH, "stopwords.txt")
    with open(stopwords_file, "r") as f:
        words = f.read()
        stop_words = [word.strip() for word in words.split(",")]
        extra_stopwords.update(set(stop_words))

    stopwords = set()
    stopwords.update(set(nltk_stopwords.words("english")))
    stopwords.update(extra_stopwords)
    stopwords.update(settings.CUSTOM_STOPWORDS)
    stopwords_list = list(stopwords)

    stopwords_dump_file = join(settings.STOPWORDS_FILE_PATH, "stopwords.pkl")
    serialize(stopwords_list, stopwords_dump_file)
    return stopwords


def get_stopwords():
    with open(join(settings.STOPWORDS_FILE_PATH, "stopwords.pkl"), "rb") as f:
        return pickle.load(f)
