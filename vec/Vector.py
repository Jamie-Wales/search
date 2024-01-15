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
        self.value = {}


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
        self.vec_normalise(self.lemmatized_data.value)
        self.vec_normalise(self.stemmed_data.value)
        self.vec_normalise(self.original_data.value)

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

        current_dic = self.__getattribute__(f"{word_type}_data").value
        passed_dic = vec.__getattribute__(
            f"{word_type}_data").value

        dot_product = sum(current_dic.get(word, 0) * passed_dic.get(word, 0) for word in common_words)

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
        for word in data.intersection:
            tf = self._tf(word_type, word)
            assert (tf >= 0)
            idf = self._idf(word_type, word)
            assert (idf >= 0)
            data.value[word] = tf * idf

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
        norm = numpy.sqrt(sum(value ** 2 for value in vec.values()))
        if norm > 0:
            for word in vec:
                vec[word] = vec[word] / norm
                assert (vec[word] >= 0)


class TFIDFVector(Vector):
    def __init__(self, corpus_word_manager: CorpusWordManager, document_word_manager: WordManager, metadata,
                 vector_space):
        super().__init__(corpus_word_manager, document_word_manager, metadata, vector_space)

    @override
    def _tf(self, word_type: str, word: str, tf=None) -> float:
        if tf is None:
            tf = self.word_manager.get_word_count(word_type, word)
        if tf == 0:
            return 0
        else:
            return math.log(tf) + 1

    @override
    def _idf(self, word_type: str, word: str) -> float:
        # Retrieve the number of documents and document frequency of the word
        num_documents = self.corpus_word_manager.number_of_documents
        doc_frequency = self.corpus_word_manager.count[word_type].get(word, 0)
        if doc_frequency == 0:
            raise ValueError

        idf = math.log(num_documents / doc_frequency + 1) + 1
        return idf


class TFIDFFieldVector(TFIDFVector):
    def __init__(self, corpus_word_manager: CorpusWordManager, document_word_manager: WordManager, metadata,
                 vector_space):
        super().__init__(corpus_word_manager, document_word_manager, metadata, vector_space)

    @override
    def _tf(self, word_type: str, word: str, tf=None) -> float:
        tags = self.word_manager.get_tag_and_count(word_type, word)
        element_weightings = {
        'metadata': 5,
        'meta': 3,
        'contenttitle': 3,
        'gameBioInfoText': 5,
        'gameBioInfo': 1,
        'gameBioHeader': 0.5,
        'gameBioInfoHeader': 0.5,
        'gameBioSysReq': 3,
        'gameBioSysReqTitle': 0.5,
        'div': 2,
        'i': 0.75,
        'strong': 1.25,
        'b': 1.25,
        'a': 2,
        'named entity': 3
        }
        tag_count_multiplier = sum(element_weightings.get(tag, 1) * count for tag, count in tags)

        tf = super()._tf(word_type, word, tag_count_multiplier)
        return tf


class BM25plusVector(Vector):
    def __init__(self, corpus_word_manager: CorpusWordManager, document_word_manager: WordManager, metadata,
                 vector_space):
        super().__init__(corpus_word_manager, document_word_manager, metadata, vector_space)

    @override
    def _tf(self, word_type: str, word: str, tf=None) -> float:
        if tf is None:
            tf = self.word_manager.get_word_count(word_type, word)
        numerator = tf * (1.2 + 1)
        doc_length = sum(sum(tag_values.values()) for tag_values in self.word_manager.words_by_tag["original"].values())

        denominator = tf + 1.2 * (1 - 0.75 + 0.75 * (doc_length / self.corpus_word_manager.avg_doc_length))
        return numerator / denominator

    @override
    def _idf(self, word_type: str, word: str) -> float:
        docs_holding_word = self.corpus_word_manager.docs_holding_word[word_type].get(word, 0)
        assert (len(docs_holding_word) != 0)
        assert (self.corpus_word_manager.number_of_documents == 399)
        numerator = self.corpus_word_manager.number_of_documents - len(docs_holding_word) + 0.5
        denominator = len(docs_holding_word) + 0.5
        idf = math.log(numerator / denominator + 1)
        assert (idf > 0)
        return idf


class BM25plusFieldVector(BM25plusVector):
    def __init__(self, corpus_word_manager: CorpusWordManager, document_word_manager: WordManager, metadata,
                 vector_space):
        super().__init__(corpus_word_manager, document_word_manager, metadata, vector_space)

    @override
    def _tf(self, word_type: str, word: str, tf=None) -> float:
        tags = self.word_manager.get_tag_and_count(word_type, word)
        element_weightings = {
            'metadata': 5,
            'meta': 3,
            'contenttitle': 3,
            'gameBioInfoText': 5,
            'gameBioInfo': 1,
            'gameBioHeader': 0.5,
            'gameBioInfoHeader': 0.5,
            'gameBioSysReq': 3,
            'gameBioSysReqTitle': 0.5,
            'div': 2,
            'i': 0.75,
            'strong': 1.25,
            'b': 1.25,
            'a': 2,
            'named entity': 3
        }

        # Calculate the tag count multiplier
        tag_count_multiplier = sum(element_weightings.get(tag, 1) * count for tag, count in tags)

        # Get the base TF from the superclass and apply the weighting
        tf = super()._tf(word_type, word, tag_count_multiplier)
        return tf


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

    def query_expansion(self, type):
        data = self.__getattribute__(f"{type}_data")
        words_to_add = []
        for word in data.intersection:
            vector_space_word = self.corpus_word_manager.get_word(type, word)
            set_of_concurrent_words = vector_space_word.__getattribute__(f"{type}_concurrent")
            for element in set_of_concurrent_words:
                print(element)
                data.value[element] = data.value.get(element, 0) + 0.25
                words_to_add.append(element)

        for word in words_to_add:
            data.intersection.add(word)

    def relevance_feedback(self, word_type: str, vec_type, relevant_doc_ids: list[Vector], beta=0.75):
        if not relevant_doc_ids:
            return {}

        shared_words = set(
            *[element.__getattribute__(f"{word_type}_data").intersection for element in relevant_doc_ids])
        print(shared_words)
        final_weighting = {}

        for word in shared_words:
            total = 0
            count = 0
            for doc in relevant_doc_ids:
                doc_data = doc.__getattribute__(f"{word_type}_data").value
                ranking = doc_data.get(word, 0)
                if ranking != 0:
                    total += ranking
                    count += 1

            final_weighting[word] = total / count if count > 0 else 0

        new_query = QueryVector(self.corpus_word_manager, self.word_manager, "", self.vector_space)

        new_query_data = new_query.__getattribute__(f"{word_type}_data")
        new_query_data.intersection.update(shared_words)

        for word in final_weighting:
            adjusted_weight = final_weighting[word] * beta
            new_query_data.value[word] = new_query_data.value.get(word, 0) + adjusted_weight

        return new_query
