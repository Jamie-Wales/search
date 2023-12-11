from concurrent.futures import ThreadPoolExecutor

from engine import Vector, TFIDFVector, TFIDFFieldVector
from utils import load, check_and_overwrite


class VectorStore:
    TFIDFVector: Vector
    TFIDFField: Vector
    BM25Vector: Vector
    BM25Field: Vector

    def __init__(self, TFIDFVector: Vector, TFIDFField: Vector):
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
            self.document_vectors = {}

    def generate_vectors(self, corpus):
        """
        Generates vectors for a given document and stores them in the document_vectors dictionary.
        """
        for document in corpus.documents:
            tfidf_vector = TFIDFVector(corpus.word_manager, document.word_manager, document.metadata, corpus.vector_space)
            tfidfField_vector = TFIDFFieldVector(corpus.word_manager, document.word_manager, document.metadata, corpus.vector_space)
            vector_store = VectorStore(tfidf_vector, tfidfField_vector)
            self.document_vectors[document.metadata.doc_id] = vector_store

        check_and_overwrite("./document-vectors.pkl", self.document_vectors)

    def get_vector(self, doc_id):
        """
        Retrieves the vector for a given document ID.
        """
        return self.document_vectors.get(doc_id)
