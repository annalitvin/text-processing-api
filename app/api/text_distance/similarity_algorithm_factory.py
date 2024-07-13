import textdistance as td

from app.api.constants import ALGORITHM_NOT_EXISTS_ERROR_MSG


class TextSimilarityAlgorithmFactory:
    _algs = td.algorithms

    @staticmethod
    def make(type):
        if type in TextSimilarityAlgorithmFactory.chose:
            return TextSimilarityAlgorithmFactory.chose[type]
        raise ValueError(ALGORITHM_NOT_EXISTS_ERROR_MSG)

    chose = {
        "hamming": _algs.hamming,
        "levenshtein": _algs.levenshtein,
        "cosine": _algs.cosine,
        "jaccard": _algs.jaccard,
    }
