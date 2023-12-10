from engine.DocumentVectorStore import DocumentVectorStore
from engine.Vector import QueryVector


class Ranker:
    def __init__(self):
        self.tf_idf_vectors = {"lemmatized": [], "stemmed": [], "original": []}

    def tf_idf_vector(self, vector_store: DocumentVectorStore, query_vec: QueryVector):
        for vec in vector_store.document_vectors.values():
            self.tf_idf_vectors["lemmatized"].append(vec.TFIDFVector.dot_product(query_vec, "lemmatized"))
            self.tf_idf_vectors["stemmed"].append(vec.TFIDFVector.dot_product(query_vec, "stemmed"))
            self.tf_idf_vectors["original"].append(vec.TFIDFVector.dot_product(query_vec, "original"))

    def tf_idf_documents(self, type):
        output = self.tf_idf_vectors[type].sort(key=lambda x: x[0], reverse=True)
        return output[:10]
