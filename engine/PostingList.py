import pickle
from collections import defaultdict
from utils import load
from typing import Dict, List

class PostingList:
    posting: Dict[str, Dict[int, List[int]]]

    def __init__(self, corpus=True):
        self.posting = self.load_posting_list("./PostingList.pkl") or defaultdict(lambda: defaultdict(list))

        if corpus and not self.posting:
            from search_components import CorpusManager
            corp_manager = CorpusManager()
            corpus = corp_manager.get_raw_corpus()
            for document in corpus.documents:
                for count, token in enumerate(document.tokenised_content):
                    self.add_posting(token, document.metadata.doc_id, count)

            # Sort posting lists
            self.sort_posting_list()
            self.save_posting_list("./PostingList.pkl")

    def add_posting(self, term: str, doc_id: int, position: int):
        self.posting[term][doc_id].append(position)

    def sort_posting_list(self):
        for doc_dict in self.posting.values():
            for positions in doc_dict.values():
                positions.sort()

    def save_posting_list(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self.posting, f)

    def load_posting_list(self, filepath):
        try:
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        except (FileNotFoundError, EOFError, pickle.UnpicklingError):
            return None


# Helper function for pickling (outside of the class)
