import heapq

from engine import DocumentVectorStore
from engine import QueryVector


class Ranker:
    @staticmethod
    def tf_idf_vector(vec_type: str, vector_store: DocumentVectorStore, query_vec: QueryVector):
        heap = []

        # Iterate through all document vectors, use enumerate to get an index
        for index, vec in enumerate(vector_store.document_vectors):
            score = vec.TFIDFVector.dot_product(query_vec, vec_type)

            # Push a tuple of (score, index, metadata) onto the heap
            if score > 0:
                heapq.heappush(heap, (score, index, vec.TFIDFVector.metadata))

            # If the heap size exceeds 10, remove the smallest element
            if len(heap) > 10:
                heapq.heappop(heap)

        # Convert heap to a list and sort in descending order
        # Using only score and metadata for the final output
        docs = sorted([(score, metadata) for score, _, metadata in heap], key=lambda x: x[0], reverse=True)
        return docs
