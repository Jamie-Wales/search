import os
import pickle
from typing import Optional, List, Dict

from search_components import Document


# TODO: maybe implement a protocol here a corpus protoco
# TODO: Clean up Corpus Manager handles generation

# manages individual corpus details
class Corpus:
    documents: List[Document]
    term_frequency: Dict[str, int]

    def __init__(self, directory_path: str):
        self.documents = []
        self.term_frequency = {}
        self.directory_path = directory_path
        self._load_documents()
        self._calculate_term_frequency()

    def _calculate_term_frequency(self):
        for document in self.documents:
            for token in document.tokenised_content:
                self.term_frequency.setdefault(token, 0)
                self.term_frequency[token] = self.term_frequency[token] + 1

    def _load_documents(self):
        from utils import DocumentParser
        parser = DocumentParser()
        # Check if directory exists
        if not os.path.exists(self.directory_path):
            raise FileNotFoundError(f"The directory {self.directory_path} does not exist")
        # Iterate over each file in the directory
        for filename in os.listdir(self.directory_path):
            filepath = os.path.join(self.directory_path, filename)
            # Check if it's a file (and not a subdirectory)
            if os.path.isfile(filepath):
                doc = parser.parse(filepath)
                self.documents.append(doc)


# singleton class to manage different types of corpus
class CorpusManager:
    _instance: Optional["CorpusManager"] = None
    raw_corpus = None

    def __init__(self) -> None:
        from utils import check_and_overwrite
        if os.path.exists("./CorpusManager.pkl"):
            file = open("./CorpusManager.pkl", "rb")
            self.raw_corpus = pickle.load(file).get_raw_corpus()
        else:
            self.raw_corpus = Corpus("./dataset/videogame/")
            check_and_overwrite("./CorpusManager.pkl", self)

    @classmethod
    def get_instance(cls) -> "CorpusManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def get_raw_corpus(self) -> Corpus:
        return self.raw_corpus

    def sort_corpus(self):
        self.raw_corpus.documents.sort(key=lambda document: document.metadata.doc_id, reverse=True)
    # todo: implement different indexing granularity
    # todo: use appropriate index granularity based on corpus type
