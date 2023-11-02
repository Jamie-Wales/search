from typing import List


class PostingList:
    def __init__(self):
        self.posting = {str: List[str]}

    def add_posting(self, term, doc_id):
       self.posting.setdefault(term, [])
       self.posting[term].append(doc_id)

       print(self.posting)
