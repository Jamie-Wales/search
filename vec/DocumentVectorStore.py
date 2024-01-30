from __future__ import annotations
from typing import Type
import numpy as np
from utils.utilities import load, check_and_overwrite


class VectorStore:
    TFIDFVector: Type["Vector"]
    TFIDFField: Type["Vector"]
    BM25Vector: Type["Vector"]
    BM25Field: Type["Vector"]

    def __init__(self, TFIDFVector: Type["Vector"], TFIDFFieldVector: Type["Vector"], BM25plusVector: Type["Vector"],
                 BM25plusFieldVector: Type["Vector"]):
        self.TFIDFVector = TFIDFVector
        self.TFIDFFieldVector = TFIDFFieldVector
        self.BM25plusVector = BM25plusVector
        self.BM25plusFieldVector = BM25plusFieldVector


class DocumentVectorStore:
    """
    This class is responsible for generating and storing vectors for each document.
    It uses the provided corpus  to compute
    the vector representations.
    """

    def __init__(self):
        self.need_vector_generation = False
        self.document_vectors = load("./pklfiles/document-vectors.pkl")
        if self.document_vectors is None:
            self.need_vector_generation = True
            self.document_vectors = np.empty(399, dtype=object)

    def generate_vectors(self, corpus):
        """
        Generates vectors for a given document and stores them in the document_vectors dictionary.
        """

        from vec.Vector import TFIDFVector, TFIDFFieldVector, BM25plusVector, BM25plusFieldVector
        for document in corpus.documents:
            tfidf_vector = TFIDFVector(corpus.word_manager, document.word_manager, document.metadata,
                                       corpus.vector_space)
            tfidf_field_vector = TFIDFFieldVector(corpus.word_manager, document.word_manager, document.metadata,
                                                  corpus.vector_space)
            bm25_vec = BM25plusVector(corpus.word_manager, document.word_manager, document.metadata,
                                      corpus.vector_space)
            bm25_field_vec = BM25plusFieldVector(corpus.word_manager, document.word_manager, document.metadata,
                                                 corpus.vector_space)
            vector_store = VectorStore(tfidf_vector, tfidf_field_vector, bm25_vec, bm25_field_vec)
            self.document_vectors.put(document.metadata.doc_id, vector_store)

        check_and_overwrite("./pklfiles/document-vectors.pkl", self.document_vectors)

    def gen_word_matrix(self, corpus):
        output = {}

        for word in corpus.word_manager.words["lemmatized"].keys():
            output[word] = {}

            # Iterate through each document vector
            for vectors in self.document_vectors:
                doc_id = vectors.BM25plusVector.metadata.doc_id
                word_value = vectors.BM25plusVector.__getattribute__(f"{"lemmatized"}_data").value.get(word, 0)
                if word_value > 0:
                    output[word][doc_id] = word_value

        return output

    def get_vector(self, doc_id) -> VectorStore:
        """
        Retrieves the vector for a given document ID.
        """
        return self.document_vectors[doc_id]
