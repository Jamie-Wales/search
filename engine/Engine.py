import math
from typing import Dict

import numpy

from engine import PostingList
from search_components import CorpusManager
from search_components.Corpus import Corpus


class Engine:
    corpus: Corpus
    posting_list: PostingList
    query_ranking: Dict[int, float]
    idf_ranking: Dict[str, float]

    def __init__(self):
        self.corpus = CorpusManager().get_raw_corpus()
        self.posting_list = PostingList()
        self.query_ranking = {}
        self.idf_ranking = {}
        self.precompute_idf()
        self.vector_space = []
        self._get_vector_space()
        self._vectorise_documents()

    def precompute_idf(self):
        n_documents = len(self.corpus.documents)
        for term in self.posting_list.posting.keys():
            self.idf_ranking[term] = self._idf(term, self.corpus.term_frequency)

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

    def _tf_idf(self, word, document_freq, term_freq):
        tf = self._tf(word, document_freq)
        idf = self._idf(word, term_freq)
        return idf * tf

    def update_ranking(self, terms):
        # Initialize a dictionary to accumulate scores for each document
        cumulative_scores = {}

        # Iterate over each term
        for term in terms:
            if term in self.idf_ranking:  # Check if IDF score is precomputed for the term
                # Calculate TF-IDF score for each document
                for document in self.corpus.documents:
                    doc_id = document.metadata.doc_id
                    score = self._tf_idf(term, document.document_frequency, self.corpus.term_frequency)

                    # If document has already been scored, accumulate the scores
                    if doc_id not in cumulative_scores:
                        cumulative_scores[doc_id] = 0
                    cumulative_scores[doc_id] += score

        # Create a list of (metadata, score) tuples for each document
        document_scores = [(doc.metadata, cumulative_scores[doc.metadata.doc_id]) for doc in self.corpus.documents if
                           doc.metadata.doc_id in cumulative_scores]

        # Sort the list of tuples by score in descending order
        document_scores.sort(key=lambda doc: doc[1], reverse=True)

        # Return the sorted list of document scores
        return document_scores

    def _get_vector_space(self):
        for tokens in self.posting_list.posting:
            self.vector_space.append(tokens)

    def _vectorise_documents(self):
        for document in self.corpus.documents:
            # Initialize an empty list to collect TF-IDF values
            tf_idf_values = []
            for token in self.corpus.term_frequency:
                # Calculate the TF-IDF value for each token and append it to the list
                tf_idf_value = self._tf_idf(token, document.document_frequency, self.corpus.term_frequency)
                tf_idf_values.append(tf_idf_value)
            # Convert the list of TF-IDF values to a numpy array and assign it to the document vector
            document.vector.raw_vec = numpy.array(tf_idf_values)