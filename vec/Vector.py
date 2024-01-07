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

        if common_words:
            for words in common_words:
                if passed_dic.get(words) == 0.25 and current_dic.get(words) > 0:
                    print(self.metadata)

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
    def _tf(self, word_type: str, word: str) -> float:
        frequency = self.word_manager.get_word_count(word_type, word)
        if frequency == 0:
            return 0
        else:
            return math.log(frequency) + 1

    @override
    def _idf(self, word_type: str, word: str) -> float:
        # Retrieve the number of documents and document frequency of the word
        num_documents = self.corpus_word_manager.number_of_documents
        doc_frequency = self.corpus_word_manager.count[word_type].get(word, 0)
        if doc_frequency == 0:
            raise ValueError

        idf = math.log(num_documents / doc_frequency)
        return idf


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
            'i': 0.75,
            'b': 1.25,
            'named entity': 3
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

    def relevance_feedback(self, word_type: str, relevant_doc_ids: list[Vector], beta=0.75):
        if not relevant_doc_ids:
            return {}

        # Create a copy of the words from the first document
        first_doc_words = relevant_doc_ids[0].__getattribute__(f"{word_type}_data").intersection
        shared_words = set(first_doc_words)
        print(shared_words)

        # Intersect with words from remaining documents to find common words
        for doc in relevant_doc_ids[1:]:
            doc_words = doc.__getattribute__(f"{word_type}_data").intersection
            shared_words.intersection_update(doc_words)

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
