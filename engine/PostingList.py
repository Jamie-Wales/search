


class PostingList:
    posting = None

    def __init__(self, corpus=True):
        # Attempt to load an existing posting list
        if self.posting is None:
            return
    def add_posting(self, term: str, doc_id: int, position: int):
        self.posting[term][doc_id].append(position)

    def sort_and_save_posting(self):
        sorted_posting = {
            token: dict(sorted(doc_dict.items(), key=lambda item: item[0]))
            for token, doc_dict in sorted(self.posting.items())
        }
        for token, doc_dict in sorted_posting.items():
            for doc_id, positions in doc_dict.items():
                positions.sort()

        check_and_overwrite("PostingList.pkl", sorted_posting)

        return sorted_posting
