from collections import defaultdict
from typing import List, Dict

from utils import inner_dict_factory, load, check_and_overwrite


class PostingList:
    posting: Dict[str, dict[int, List[int]]]

    # url,STRING : esrb,STRING : publisher,STRING : genre,STRING : developer
    def __init__(self, corpus=True):
        if corpus:
            PostingList = load("PostingList.pkl")
            if PostingList is None:
                self.posting = defaultdict(inner_dict_factory)
                from search_components import CorpusManager
                corp_manager = CorpusManager()
                corpus = corp_manager.get_raw_corpus()
                for document in corpus.documents:
                    for count, token in enumerate(document.tokenised_content):
                        if document.metadata:
                            self.add_posting(token, document.metadata.doc_id, count)

                sorted_posting = {
                    token: dict(sorted(doc_dict.items(), key=lambda item: item[0]))
                    for token, doc_dict in sorted(self.posting.items())
                }
                self.posting = sorted_posting
                for token, doc_dict in self.posting.items():
                    for doc_id, positions in doc_dict.items():
                        positions.sort()

                check_and_overwrite("PostingList.pkl", self)
            else:
                self.posting = PostingList.posting

    def add_posting(self, term: str, doc_id: int, count: int):
        self.posting[term][doc_id].append(count)

    def add_posting_list(self, term: List[str], doc_id: int, lem=False, stem=True):
        if doc_id is not None:
            count = 0
            for terms in term:
                if stem is True:
                    pass
                if lem is True:
                    pass

                self.posting[terms][doc_id].append(count)

    def load_posting_list(self, url):
        pl = load(url)
        self.posting = pl.posting
