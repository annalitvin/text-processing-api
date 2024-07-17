from os import path
from typing import Set

from pydantic import BaseConfig


PARENT_DIR = path.dirname(path.abspath(__file__))


class TextPreprocessingConfig(BaseConfig):

    stopwords_file_path: str = path.join(PARENT_DIR, "stopwords")
    raw_stopwords_file_path: str = path.join(stopwords_file_path, "raw_stopwords")

    custom_stopwords: Set[str] = {"http", "https", "com"}

    raw_text_file_path = path.join(PARENT_DIR, "raw_text")

    preprocessing_result_file_path = path.join(PARENT_DIR, "preprocessing_result")

    class Config:
        case_sensitive = True


config = TextPreprocessingConfig()
