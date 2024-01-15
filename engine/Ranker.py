from __future__ import annotations

import heapq


class Ranker:
    @staticmethod
    def tf_idf_vector(word_type: str, vec_type: str, vector_store: "DocumentVectorStore", query_vec: "QueryVector", ner_words):
        heap = []
        for index, vec in enumerate(vector_store.document_vectors):
            vector = vec.__getattribute__(vec_type)
            score = vector.dot_product(query_vec, word_type)

            if score > 0 and len(ner_words) > 0:
                for word in ner_words.values():
                    check_word = vector.word_manager.words["original"].get(f"{word[0]}, {word[1]}", None)
                    if check_word is not None and check_word.type == word[1]:
                        print("named entity")
                        score *= 2

            if score > 0:
                heapq.heappush(heap, [score, index, vector.metadata,
                                      vector.__getattribute__(f"{word_type}_data").intersection.intersection(
                                          query_vec.__getattribute__(f"{word_type}_data").intersection)])

            if len(heap) > 10:
                heapq.heappop(heap)

        docs = sorted([[score, metadata, intersection] for score, _, metadata, intersection in heap],
                      key=lambda x: x[0], reverse=True)
        return docs
