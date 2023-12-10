import math
from typing import Dict, List

import numpy

from search_components import CorpusManager, Corpus
from utils import DocumentProcessor


class Engine:
    doc_length: float
    corpus: Corpus
    query_ranking: Dict[int, float]

    def __init__(self):

        self.corpus = CorpusManager().get_raw_corpus()
        self.vector_space = []
        self.doc_length = self._avg_doc_length()
        self._get_vector_space()
        self._vectorise_documents()
        self.word_doc_matrix = self.get_word_doc_matrix()

    def _tf(self, word, document_frequency: dict[str, int]):
        freq = document_frequency.get(word, 0)
        return math.log(freq + 1)

    def _idf(self, word) -> float:
        n_documents = self.corpus.term_frequency.get(word)
        return math.log((len(self.corpus.documents)) / (n_documents + 1) + 1)

    def _query_tf(self, word: str, query_list: List[str]):
        count = 0
        for query in query_list:
            if word is query:
                count = count + 1
        return count

    def _tf_idf(self, word, document_freq, term_freq, query=False, query_list=None, bmf=False, metadata=None,
                field=True, sections=None):
        if query:
            return self._query_tf(word, query_list)
        idf = self._idf(word)

        if bmf:
            return self.okampi25plusidf(word) * self.okampi25plus(word, term_freq, document_freq, metadata=metadata,
                                                                  field=field, sections=sections)
        else:
            if field:
                ## If a query is 1 and then normalised
                score = self.field(word, metadata, sections)
                tf = self._tf(word, document_freq)
                return tf * (score * idf)
            else:
                tf = self._tf(word, document_freq)
                return tf * idf

    def update_ranking(self, query_vec):
        document_score = []
        for document in self.corpus.documents:
            score = 0
            for term, value in document.vector.raw_vec.items():
                if term in query_vec:
                    score += value * query_vec[term]
            document_score.append((document.metadata, score))

        document_score.sort(key=lambda doc: doc[1], reverse=True)
        return document_score

    def _get_vector_space(self):
        for tokens in self.corpus.word_manager.words:
            self.vector_space.append(tokens)

    def _vectorise_documents(self, bm25f=True):
        for document in self.corpus.documents:
            # Calculate the intersection of terms in the document and the vector space
            doc_terms = set(document.document_frequency.keys())
            corpus_terms = set(self.vector_space)
            intersection = doc_terms.intersection(corpus_terms)

            # Initialize a dictionary for storing TF-IDF values
            tf_idf_values = {}

            for token in intersection:

                if bm25f:
                    # Calculate the TF-IDF value for each token in the intersection
                    tf_idf_value = self._tf_idf(token, document.document_frequency, self.corpus.term_frequency,
                                                metadata=document.metadata, sections=document.tags_to_words)
                else:
                    tf_idf_value = self._tf_idf(token, document.document_frequency, self.corpus.term_frequency,
                                                metadata=document.metadata, sections=document.tags_to_words,
                                                field=True)
                tf_idf_values[token] = tf_idf_value

            # Store the sparse representation
            document.vector.raw_vec = tf_idf_values
            document.vector.intersection = intersection

            self.vec_normalise(document.vector.raw_vec)

    def vec_normalise(self, vec):
        # Normalize the sparse vector
        norm = numpy.sqrt(sum(value ** 2 for value in vec.values()))
        if norm > 0:
            for token in vec:
                vec[token] /= norm

    def vectorize_query(self, query_terms: List[str]):
        query_vec = {}
        for term in query_terms:
            if term in self.vector_space:
                tf_idf_value = self._tf_idf(term, self.corpus.term_frequency, query_terms, query_list=query_terms,
                                            query=True)
                query_vec[term] = tf_idf_value
        return query_vec

    def process_query(self, terms: list[str]):
        vec = self.vectorize_query(terms)
        self.vec_normalise(vec)
        return self.update_ranking(vec)

    def field(self, word, metadata, sections):
        element_weightings = {
            'title': 5,
            'table': 2,
            # Assigning a weight of 1 to other elements as per your instruction
            'head': 1,
            'span': 1,
            'b': 1,
            'i': 1,
            'form': 1,
            'p': 1,
            'div': 1,
        }

        dp = DocumentProcessor()

        metadata_count = 0
        metadata_score = 0
        a, b, c, d = dp.tokenise(metadata.esrb), dp.tokenise(metadata.genre), dp.tokenise(
            metadata.developer), dp.tokenise(metadata.publisher)

        for tokens in [a, b, c, d]:
            if word in tokens:
                metadata_count += 1
                metadata_score += 5  # Weight for metadata is 5

        element_score = 0
        for weight in element_weightings.keys():
            tag_count = sections.get(weight, {}).get(word, 0)
            element_score += element_weightings[weight] * tag_count

        total_score = metadata_score + element_score
        return total_score

    def okampi25plus(self, word, term_frequency, document_frequency, metadata, field=True, sections=None):
        if field:
            total_score = self.field(word, metadata, sections)
            tf = total_score * (1.2 + 1) / (
                    total_score + 1.2 * (1 - 0.75 + 0.75 * sum(document_frequency.values()) / self.doc_length))
            return tf

    def _avg_doc_length(self):
        count = 0
        sum = 0
        for doc in self.corpus.documents:
            sum += len(doc.document_frequency)
            count += 1

        return sum / count

    def okampi25plusidf(self, word):
        docsHoldingN = self._get_n_docs(word)
        top = len(self.corpus.documents) - docsHoldingN
        idf = math.log(top + 0.5 / (docsHoldingN + 0.5))
        return idf

    def _get_n_docs(self, word):
        count = 0
        for docs in self.corpus.documents:
            if word in docs.document_frequency:
                count += 1
        return count

    def get_word_doc_matrix(self):
        # Initialize a dictionary where each word maps to a list of TF-IDF scores, one per document
        word_score = {}
        for word in self.vector_space:
            word_score[word] = []
            for doc in self.corpus.documents:
                word_score[word].append(doc.vector.raw_vec.get(word))

        return word_score

    def find_most_related_word(self, target_word):
        sorted_related_words = []
        word_arr = self.word_doc_matrix.get(target_word)

        # Ensure word_arr is not None before proceeding
        if word_arr is not None:
            for keys in self.word_doc_matrix.keys():
                check = self.word_doc_matrix[keys]
                score = 0
                # Ensure check is not None before doing dot product
                if check is not None:
                    check = [0 if v is None else v for v in check]
                    word_arr = [0 if v is None else v for v in word_arr]
                    score += numpy.dot(check, word_arr)
                    sorted_related_words.append((keys, score))

            sorted_related_words = sorted(sorted_related_words, key=lambda x: x[1], reverse=True)
            top_5_words = [word for word, score in sorted_related_words if word != target_word][:5]

            return top_5_words
