from collections import defaultdict

from utils import load, check_and_overwrite, inner_dict_factory


class PostingList:
    posting = None

    def __init__(self, corpus=True):
        # Attempt to load an existing posting list
        self.posting = load("PostingList.pkl")
        if self.posting is None:
            self.posting = defaultdict(inner_dict_factory)
            if corpus:
                from search_components import CorpusManager
                corp_manager = CorpusManager()
                corpus = corp_manager.get_raw_corpus()
                for document in corpus.documents:
                    for count, token in enumerate(document.tokenised_content):
                        if document.metadata:
                            self.add_posting(token, document.metadata.doc_id, count)

                # Sort and save the posting
                self.posting = self.sort_and_save_posting()

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
