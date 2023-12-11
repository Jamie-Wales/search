import math

import numpy

from engine import Search


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


if __name__ == "__main__":
    search = Search().search("grand theft auto")