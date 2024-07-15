import textdistance as td

from enum import Enum, unique

from app.api.constants import ALGORITHM_NOT_EXISTS_ERROR_MSG


@unique
class AlgorithmType(Enum):
    HAMMING = "hamming"
    LEVENSTEIN = "levenshtein"
    COSINE = "cosine"
    JACCARD = "jaccard"


class TextSimilarityAlgorithmFactory:
    _algs = td.algorithms

    @staticmethod
    def make(type):
        if type in TextSimilarityAlgorithmFactory.chose:
            return TextSimilarityAlgorithmFactory.chose[type]
        raise ValueError(ALGORITHM_NOT_EXISTS_ERROR_MSG)

    chose = {
        AlgorithmType.HAMMING.value: _algs.hamming,
        AlgorithmType.LEVENSTEIN.value: _algs.levenshtein,
        AlgorithmType.COSINE.value: _algs.cosine,
        AlgorithmType.JACCARD.value: _algs.jaccard,
    }
