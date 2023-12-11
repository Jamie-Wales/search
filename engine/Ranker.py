from engine import DocumentVectorStore
from engine import QueryVector


class Ranker:
    def __init__(self):
        self.tf_idf_vectors = {"lemmatized": [], "stemmed": [], "original": []}

    def tf_idf_vector(self, vector_store: DocumentVectorStore, query_vec: QueryVector):
        for vec in vector_store.document_vectors.items():
            self.tf_idf_vectors["lemmatized"].append((vec[1].TFIDFVector.dot_product(query_vec, "lemmatized"), vec[0]))
            self.tf_idf_vectors["stemmed"].append((vec[1].TFIDFVector.dot_product(query_vec, "stemmed"), vec[0]))
            self.tf_idf_vectors["original"].append((vec[1].TFIDFVector.dot_product(query_vec, "original"), vec[0]))

    def tf_idf_documents(self, type):
        self.tf_idf_vectors[type].sort(key=lambda x: x[0], reverse=True)
