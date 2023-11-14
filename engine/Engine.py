import math
from typing import Dict, List

import numpy

from engine import PostingList
from search_components import CorpusManager, Corpus


class Engine:
    corpus: Corpus
    posting_list: PostingList
    query_ranking: Dict[int, float]
    idf_ranking: Dict[str, float]

    def __init__(self):
        self.corpus = CorpusManager().get_raw_corpus()
        self.query_ranking = {}
        self.idf_ranking = {}
        self.posting_list = PostingList()
        self.vector_space = []
        self._get_vector_space()
        self._vectorise_documents()

    def _tf(self, word, document_frequency: dict[str, int]):
        freq = document_frequency.get(word, 0)
        return math.log10(freq + 1)

    def _idf(self, word: str, document_frequency):
        n_documents = self._n_containing(word, self.corpus)
        return math.log10((len(self.corpus.documents)) / (n_documents + 1) + 1)

    @staticmethod
    def _n_containing(word: str, corpus: Corpus):
        count = 0
        for docs in corpus.documents:
            word_exists = docs.document_frequency.get(word, 0)
            if word_exists != 0:
                count = count + 1
        return count

    def _query_tf(self, word: str, query_list: List[str]):
        count = 0
        for query in query_list:
            if word is query:
                count = count + 1
        return count

    def _tf_idf(self, word, document_freq, term_freq, query=False, query_list=None):
        if not query:
            tf = self._tf(word, document_freq)
        else:
            tf = self._query_tf(word, query_list)
        idf = self._idf(word, term_freq)
        return tf * idf

    def update_ranking(self, vec):
        document_score = []
        for document in self.corpus.documents:
            document_score.append((document.metadata, document.vector.raw_vec.dot(vec.raw_vec)))

        document_score.sort(key=lambda doc: doc[1], reverse=True)

        # Return the sorted list of document scores
        return document_score

    def _get_vector_space(self):
        for tokens in self.posting_list.posting:
            self.vector_space.append(tokens)

    def _vectorise_documents(self):
        for document in self.corpus.documents:
            # Initialize a numpy array with zeros, the size of the vector space
            tf_idf_values = numpy.zeros(len(self.vector_space))

            for index, token in enumerate(self.vector_space):
                # Calculate the TF-IDF value for each token
                tf_idf_value = self._tf_idf(token, document.document_frequency, self.corpus.term_frequency)
                # Place the TF-IDF value directly in the numpy array
                tf_idf_values[index] = tf_idf_value

            # The numpy array is already initialized, no need to convert
            document.vector.raw_vec = tf_idf_values
            self.vec_normalise(document.vector)

    def vec_normalise(self, vec):
        vec.raw_vec = vec.raw_vec / numpy.sqrt(numpy.sum(vec.raw_vec ** 2))

    def vectorize_query(self, query_terms: List[str], vector_space: List[str],
                        corpus_tf: Dict[str, int]) -> numpy.array:
        from engine import Vector
        output = Vector()
        query_vector = numpy.zeros(len(self.vector_space))
        for term in query_terms:
            if term in vector_space:
                term_index = vector_space.index(term)
                # tidy this up jamie
                tf_idf_value = self._tf_idf(term, query_terms, corpus_tf, query_list=query_terms,
                                            query=True)  # Implement this function
                query_vector[term_index] = tf_idf_value
        output.raw_vec = query_vector
        return output

    def process_query(self, terms: list[str]):
        vec = self.vectorize_query(terms, self.vector_space, self.corpus.term_frequency)
        self.vec_normalise(vec)
        return self.update_ranking(vec)

    def okampi25plus(self, word, term_frequency, document_frequency):
        return ((term_frequency[word] * (1.2 + 1)) / (term_frequency[word] + 1.2) *
                (1 - 0.75 + 0.75 * (len(document_frequency) / self._avg_doc_length())))


    def _avg_doc_length(self):
        count = 0
        sum = 0
        for doc in self.corpus.documents:
            sum += len(doc.document_frequency)
            count += 1

        return sum / count