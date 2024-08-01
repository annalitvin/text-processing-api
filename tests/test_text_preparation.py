import pytest

from nltk.tree.tree import Tree


class TestTextPreparation:

    @pytest.mark.parametrize(
        "expected",
        [["review", "summar", "summar", "review"]],
    )
    def test_stemmed(self, text_preparation, expected):
        assert text_preparation.stemmed() == expected

    @pytest.mark.parametrize(
        "expected",
        [
            ["review", "summarizing", "summarizing", "review"],
        ],
    )
    def test_lemmatize(self, text_preparation, expected):
        assert text_preparation.lemmatize() == expected

    @pytest.mark.parametrize(
        "expected",
        [
            Tree(
                "S",
                [
                    ("a", "DT"),
                    ("review", "NN"),
                    ("an", "DT"),
                    ("summarizing", "VBG"),
                    ("the", "DT"),
                    ("of", "IN"),
                    ("the", "DT"),
                    ("the", "DT"),
                    ("the", "DT"),
                    ("and", "CC"),
                    ("a", "DT"),
                    ("summarizing", "VBG"),
                    ("the", "DT"),
                    ("review", "NN"),
                    ("an", "DT"),
                    ("of", "IN"),
                    ("the", "DT"),
                    ("and", "CC"),
                    ("the", "DT"),
                ],
            ),
        ],
    )
    def test_extract_parts_of_speech(self, text_preparation, expected):
        assert text_preparation.extract_parts_of_speech() == expected

    @pytest.mark.parametrize(
        "expected",
        [
            {"neg": 0.0, "neu": 1.0, "pos": 0.0, "compound": 0.0},
        ],
    )
    def test_sentiment_analysis(self, text_preparation, expected):
        assert text_preparation.sentiment_analysis() == expected
