import math
from abc import ABC, abstractmethod
from typing import override

import numpy

from search_components.WordManager import WordManager, CorpusWordManager


class VectorData:
    def __init__(self, raw_list, intersection):
        self.raw_list = raw_list
        self.intersection = intersection
        self.value = numpy.zeros(len(intersection))


class VectorSpace:
    def __init__(self, corpus_word_manager: CorpusWordManager):
        self.lemmatized_vectorspace = numpy.array(list(corpus_word_manager.words['lemmatized'].keys()))
        self.stemmed_vectorspace = numpy.array(list(corpus_word_manager.words['stemmed'].keys()))
        self.original_vectorspace = numpy.array(list(corpus_word_manager.words['original'].keys()))


class Vector(ABC):
    def __init__(self, corpus_word_manager: CorpusWordManager, word_manager: WordManager, doc_id):
        self.doc_id = doc_id
        self.corpus_word_manager = corpus_word_manager
        self.word_manager = word_manager
        self.vector_space = VectorSpace(corpus_word_manager)

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
        # Combine all unique words from both intersections
        current_vec_data = self.__getattribute__(f"{word_type}_data").intersection
        passed_vec_data = vec.__getattribute__(f"{word_type}_data").intersection
        all_words = current_vec_data.union(passed_vec_data)

        # For each word in the union of both intersections,
        # we take the corresponding value from self.value,
        # or 0 if the word is not in self.intersection
        self_vector = [current_vec_data[passed_vec_data.index(word)] if word in current_vec_data else 0 for
                       word in all_words]

        # Do the same for vec
        vec_vector = [
            passed_vec_data.value[passed_vec_data.raw_list.index(word)] if word in passed_vec_data.raw_list else 0 for
            word in all_words]

        # Calculate the dot product
        dot_product = sum(a * b for a, b in zip(self_vector, vec_vector))

        return dot_product

    @abstractmethod
    def weightingAlgorithm(self) -> None:
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
    def __init__(self, corpus_word_manager: CorpusWordManager, document_word_manager: WordManager, doc_id):
        super().__init__(corpus_word_manager, document_word_manager, doc_id)

    @override
    def weightingAlgorithm(self) -> None:
        lemmatized_data = self.lemmatized_data
        for index, word in enumerate(lemmatized_data.intersection):
            lemmatized_data.value[index] = (self._tf("lemmatized", word, self.word_manager) *
                                            self._idf("lemmatized", word, self.corpus_word_manager))
        stemmed_data = self.stemmed_data
        for index, word in enumerate(stemmed_data.intersection):
            stemmed_data.value[index] = (self._tf("stemmed", word, self.word_manager) *
                                         self._idf("stemmed", word, self.corpus_word_manager))
        original_data = self.original_data
        for index, word in enumerate(original_data.intersection):
            original_data.value[index] = (self._tf("original", word, self.word_manager) *
                                          self._idf("original", word, self.corpus_word_manager))

    @staticmethod
    def _tf(word_type: str, word: str, document_word_manager: WordManager) -> float:
        frequency = document_word_manager.get_word_count(word_type, word)
        return math.log(frequency + 1)

    @staticmethod
    def _idf(word_type: str, word: str, corpus_word_manager: CorpusWordManager) -> float:
        return math.log(
            corpus_word_manager.number_of_documents / corpus_word_manager.n_doc_count.get(word_type).get(word, 1) + 1)


class TFIDFFieldVector(TFIDFVector):
    def __init__(self, corpus_word_manager: CorpusWordManager, document_word_manager: WordManager, doc_id):
        super().__init__(corpus_word_manager, document_word_manager, doc_id)

    @staticmethod
    @override
    def _tf(word_type: str, word: str, document_word_manager: WordManager) -> float:
        tags = document_word_manager.get_tag_and_count(word_type, word)
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
    def __init__(self, corpus_word_manager: CorpusWordManager, document_word_manager: WordManager, doc_id):
        super().__init__(corpus_word_manager, document_word_manager, doc_id)

    def weightingAlgorithm(self) -> None:
        pass


class BM25plusFieldVector(Vector):
    def __init__(self, corpus_word_manager: CorpusWordManager, document_word_manager: WordManager, doc_id):
        super().__init__(corpus_word_manager, document_word_manager, doc_id)

    def weightingAlgorithm(self) -> None:
        pass



class QueryVector(TFIDFVector):
    def __init__(self, corpus_word_manager: CorpusWordManager, document_word_manager: WordManager):
        super().__init__(corpus_word_manager, document_word_manager, doc_id="query")

    def weightingAlgorithm(self) -> float:
        lemmatized_data = self.lemmatized_data
        for index, word in enumerate(lemmatized_data.intersection):
            lemmatized_data.value[index] = (self._tf("lemmatized", word, self.word_manager))

        stemmed_data = self.stemmed_data
        for index, word in enumerate(stemmed_data.intersection):
            stemmed_data.value[index] = (self._tf("stemmed", word, self.word_manager))
        original_data = self.original_data
        for index, word in enumerate(original_data.intersection):
            original_data.value[index] = (self._tf("original", word, self.word_manager))