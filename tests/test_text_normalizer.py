import pytest


class TestTextNormalizer:

    @pytest.mark.parametrize(
        "expected",
        [
            (
                "a traditional critical review includes an introduction summarizing the key "
                "details of the work being reviewed the body containing the evaluation and "
                "a conclusion summarizing the review instructional texts usually start "
                "with an overview of the task or goal and possibly what the end result should look like"
            ),
        ],
    )
    def test_cleaned_text(self, text_normalizer, expected):
        assert text_normalizer.text == expected

    @pytest.mark.parametrize(
        "expected",
        [
            [
                "traditional",
                "critical",
                "review",
                "includes",
                "introduction",
                "summarizing",
                "key",
                "details",
                "work",
                "reviewed",
                "body",
                "evaluation",
                "conclusion",
                "summarizing",
                "review",
                "instructional",
                "texts",
                "start",
                "overview",
                "task",
                "goal",
                "result",
            ],
        ],
    )
    def test_normalize_no_stopwords(self, text_normalizer, expected):
        normalized_words = text_normalizer.normalize(del_stopwords=True)
        assert normalized_words == expected

    @pytest.mark.parametrize(
        "expected",
        [
            [
                "a",
                "traditional",
                "critical",
                "review",
                "includes",
                "an",
                "introduction",
                "summarizing",
                "the",
                "key",
                "details",
                "of",
                "the",
                "work",
                "being",
                "reviewed",
                "the",
                "body",
                "containing",
                "the",
                "evaluation",
                "and",
                "a",
                "conclusion",
                "summarizing",
                "the",
                "review",
                "instructional",
                "texts",
                "usually",
                "start",
                "with",
                "an",
                "overview",
                "of",
                "the",
                "task",
                "or",
                "goal",
                "and",
                "possibly",
                "what",
                "the",
                "end",
                "result",
                "should",
                "look",
                "like",
            ],
        ],
    )
    def test_normalize_with_stopwords(self, text_normalizer, expected):
        normalized_words = text_normalizer.normalize(del_stopwords=False)
        assert normalized_words == expected

    @pytest.mark.parametrize(
        "expected",
        [
            "a review an summarizing the of the the the and a summarizing the review an of the and the",
        ],
    )
    def test_normalize_remove_rare_words(self, text_normalizer, expected):
        text_normalizer.normalize(del_stopwords=False, remove_rare_words=True)
        assert text_normalizer.text == expected
