from typing import List, Dict


class DocumentMetaData:
    esrb: str
    publisher: str
    genre: str
    developer: str

    def __init__(self, esrb, publisher, genre, developer):
        self.esrb = esrb
        self.publisher = publisher
        self.genre = genre
        self.developer = developer


class PostingList:
    document_map: Dict[str, int]
    posting: Dict[str, List[str]]

    # url,STRING : esrb,STRING : publisher,STRING : genre,STRING : developer
    def __init__(self, path="./dataset/video-game-labels.csv"):
        self.posting = {}
        with open(path, mode="r") as f:
            for line in f.readlines():
                csv = line.split(",")

    def add_posting(self, term: str, doc_id: str):
        self.posting.setdefault(term, [])
        self.posting[term].append(doc_id)
