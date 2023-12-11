import math
from abc import ABC, abstractmethod
from threading import Thread
from typing import override, Callable

import numpy

from search_components.WordManager import WordManager, CorpusWordManager


class VectorData:
    def __init__(self, raw_list, intersection):
        self.raw_list = raw_list
        self.intersection = intersection
        self.value = numpy.zeros(len(intersection))


class Vector(ABC):
    def __init__(self, corpus_word_manager: CorpusWordManager, word_manager: WordManager, metadata,
                 vector_space):
        self.metadata = metadata
        self.corpus_word_manager = corpus_word_manager
        self.word_manager = word_manager
        self.vector_space = vector_space

        self.lemmatized_data = self._process_vector_data("lemmatized")
        self.stemmed_data = self._process_vector_data("stemmed")
        self.original_data = self._process_vector_data("original")

        self.weightingAlgorithm()
        self.lemmatized_data.value = self.vec_normalise(self.lemmatized_data.value)
        self.stemmed_data.value = self.vec_normalise(self.stemmed_data.value)
        self.original_data.value = self.vec_normalise(self.original_data.value)

    def _process_vector_data(self, word_type) -> VectorData:
        raw_vec = self._generate_raw_vec(self.word_manager.words[word_type],
                                         self.vector_space.__getattribute__(f"{word_type}_vectorspace"))
        intersection = self._getIntersection(raw_vec, self.vector_space.__getattribute__(f"{word_type}_vectorspace"))
        vector_data = VectorData(raw_vec, intersection)
        return vector_data

    def dot_product(self, vec: 'Vector', word_type) -> float:
        # Intersection of words in both vectors
        common_words = self.__getattribute__(f"{word_type}_data").intersection & vec.__getattribute__(
            f"{word_type}_data").intersection

        # Create dictionaries from intersections and values
        current_values_dict = dict(zip(self.__getattribute__(f"{word_type}_data").intersection,
                                       self.__getattribute__(f"{word_type}_data").value))
        passed_values_dict = dict(zip(vec.__getattribute__(f"{word_type}_data").intersection,
                                      vec.__getattribute__(f"{word_type}_data").value))

        # Calculate the dot product for common words
        dot_product = sum(current_values_dict.get(word, 0) * passed_values_dict.get(word, 0) for word in common_words)

        return dot_product

    @staticmethod
    def parallel_weighting(weighting_function: Callable, *args):
        threads = []
        for word_type in ['lemmatized', 'stemmed', 'original']:
            # Create a process for each word type
            thread = Thread(target=weighting_function, args=(word_type, *args))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

    def weightingAlgorithm(self) -> None:
        self.parallel_weighting(self._weighting_for_type)

    def _weighting_for_type(self, word_type):
        data = self.__getattribute__(f"{word_type}_data")
        for index, word in enumerate(data.intersection):
            tf = self._tf(word_type, word)
            idf = self._idf(word_type, word)
            data.value[index] = tf * idf

    @abstractmethod
    def _tf(self, word_type: str, word: str) -> float:
        pass

    @abstractmethod
    def _idf(self, word_type: str, word: str) -> float:
        pass

    @staticmethod
    def _generate_raw_vec(word_list, vector_space_type):
        raw_vec = numpy.array([word if word_list.get(word) else 0 for word in vector_space_type])
        return raw_vec

    @staticmethod
    def _getIntersection(raw_vec, vector_space_type):
        vector_space_set = set(vector_space_type)
        raw_vec_set = set(raw_vec)
        intersection = vector_space_set.intersection(raw_vec_set)
        return intersection

    @staticmethod
    def vec_normalise(vec):
        output = []
        # Normalize the sparse vector
        norm = numpy.sqrt(sum(value ** 2 for value in vec))
        if norm > 0:
            for value in vec:
                value /= norm
                output.append(value)
        return output


class TFIDFVector(Vector):
    def __init__(self, corpus_word_manager: CorpusWordManager, document_word_manager: WordManager, metadata,
                 vector_space):
        super().__init__(corpus_word_manager, document_word_manager, metadata, vector_space)

    @override
    def _tf(self, word_type: str, word: str) -> float:
        frequency = self.word_manager.get_word_count(word_type, word)
        return 1 + math.log(frequency) if frequency > 0 else 0

    @override
    def _idf(self, word_type: str, word: str) -> float:
        # Specific IDF implementation for TFIDFVector
        return math.log(
            self.corpus_word_manager.number_of_documents /
            self.corpus_word_manager.n_doc_count.get(word_type).get(word, 1))


class TFIDFFieldVector(TFIDFVector):
    def __init__(self, corpus_word_manager: CorpusWordManager, document_word_manager: WordManager, metadata,
                 vector_space):
        super().__init__(corpus_word_manager, document_word_manager, metadata, vector_space)

    @override
    def _tf(self, word_type: str, word: str) -> float:
        tags = self.word_manager.get_tag_and_count(word_type, word)
        frequency = 0
        element_weightings = {
            'meta': 3,
            'contenttitle': 2.5,
            'gameBioInfoText': 2.25,
            'gameBioInfo': 1.75,
            'gameBioHeader': 0.5,
            'gameBioInfoHeader': 0.5,
            'gameBioSysReq': 1.25,
            'gameBioSysReqTitle': 0.5,
            'div': 1,
        }

        for tag in tags:
            frequency = element_weightings[tag[0]] * tag[1]

        return math.log(frequency + 1)


class BM25plusVector(Vector):
    def __init__(self, corpus_word_manager: CorpusWordManager, document_word_manager: WordManager, metadata,
                 vector_space):
        super().__init__(corpus_word_manager, document_word_manager, metadata, vector_space)

    def weightingAlgorithm(self) -> None:
        pass


class BM25plusFieldVector(Vector):
    def __init__(self, corpus_word_manager: CorpusWordManager, document_word_manager: WordManager, metadata,
                 vector_space):
        super().__init__(corpus_word_manager, document_word_manager, metadata, vector_space)

    def weightingAlgorithm(self) -> None:
        pass


class QueryVector(Vector):
    def __init__(self, corpus_word_manager: CorpusWordManager, document_word_manager: WordManager, metadata,
                 vector_space):
        super().__init__(corpus_word_manager, document_word_manager, metadata, vector_space)

    @override
    def _tf(self, word_type: str, word: str) -> int:
        return self.word_manager.get_word_count(word_type, word)

    @override
    def _idf(self, word_type: str, word: str) -> int:
        return 1
