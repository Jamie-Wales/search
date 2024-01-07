from __future__ import annotations

from typing import Type

import numpy as np
from utils.utilities import load, check_and_overwrite


class VectorStore:
    TFIDFVector: Type["Vector"]
    TFIDFField: Type["Vector"]
    BM25Vector: Type["Vector"]
    BM25Field: Type["Vector"]

    def __init__(self, TFIDFVector: Type["Vector"], TFIDFField: Type["Vector"]):
        self.TFIDFVector = TFIDFVector
        self.TFIDFField = TFIDFField


class DocumentVectorStore:
    """
    This class is responsible for generating and storing vectors for each document.
    It uses the provided corpus word manager and document word manager to compute
    the vector representations.
    """

    def __init__(self):
        self.need_vector_generation = False
        self.document_vectors = load("./document-vectors.pkl")
        if self.document_vectors is None:
            self.need_vector_generation = True
            self.document_vectors = np.empty(399, dtype=object)

    def generate_vectors(self, corpus):
        """
        Generates vectors for a given document and stores them in the document_vectors dictionary.
        """

        from vec.Vector import TFIDFVector, TFIDFFieldVector
        for document in corpus.documents:
            tfidf_vector = TFIDFVector(corpus.word_manager, document.word_manager, document.metadata,
                                       corpus.vector_space)
            tfidfField_vector = TFIDFFieldVector(corpus.word_manager, document.word_manager, document.metadata,
                                                 corpus.vector_space)
            vector_store = VectorStore(tfidf_vector, tfidfField_vector)
            self.document_vectors.put(document.metadata.doc_id, vector_store)

        check_and_overwrite("./document-vectors.pkl", self.document_vectors)

    def gen_word_matrix(self, corpus):
        output = {}

        # Iterate through each word type (e.g., 'original', 'stemmed', 'lemmatized')
        for type in corpus.word_manager.words.keys():
            output[type] = {}

            # Iterate through each word in the word type
            for word in corpus.word_manager.words[type].keys():
                output[type][word] = {}

                # Iterate through each document vector
                for vectors in self.document_vectors:
                    doc_id = vectors.TFIDFVector.metadata.doc_id
                    word_value = vectors.TFIDFVector.__getattribute__(f"{type}_data").value.get(word, 0)
                    if word_value > 0:
                        output[type][word][doc_id] = word_value

        return output

    def get_vector(self, doc_id) -> VectorStore:
        """
        Retrieves the vector for a given document ID.
        """
        return self.document_vectors[doc_id]
