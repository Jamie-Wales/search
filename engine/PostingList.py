from typing import List


class PostingList:
    def __init__(self):
        self.posting = {str: List[str]}

    def add_posting(self, term, doc_id):
        if term not in self.posting:
            self.posting[term] = [doc_id]
        else:
            self.posting[term].append(doc_id)
