import os
import pickle
from typing import Optional, List

from search_components import Document
from search_components.WordManager import WordManager


# manages individual corpus details
class Corpus:
    documents: List[Document]
    word_manager: WordManager

    def __init__(self, directory_path: str):
        self.documents = []
        self.directory_path = directory_path
        self._load_documents()
        self.word_manager = WordManager()
        self.word_manager.from_document_managers(self.documents)

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

    def sort_corpus(self):
        self.documents.sort(key=lambda document: document.metadata.doc_id, reverse=True)

    # todo: implement different indexing granularity
    # todo: use appropriate index granularity based on corpus type


class CorpusManager:
    _instance: Optional["CorpusManager.pkl"] = None
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
