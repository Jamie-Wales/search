import math
from typing import Dict

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

    def precompute_idf(self):
        n_documents = len(self.corpus.documents)
        for term in self.posting_list.posting.keys():
            self.idf_ranking[term] = self._idf(term, self.corpus.term_frequency)

    def _tf(self, word, document_frequency: dict[str, int]):
        if word not in document_frequency:
            return 0
        else:
            return document_frequency[word] / sum(document_frequency.values())

    @staticmethod
    def _n_containing(word: str, term_frequency: dict[str, int]):
        return term_frequency[word]

    def _idf(self, word: str, document_frequency):
        n_documents = len(self.corpus.documents)
        return math.log10(n_documents / (1 + document_frequency.get(word, 0)))

    def _tf_idf(self, word, document_freq, term_freq):
        return self._tf(word, term_freq) * self._idf(word, document_freq)

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
        document_scores.sort(key=lambda doc: doc[1])

        # Return the sorted list of document scores
        return document_scores


