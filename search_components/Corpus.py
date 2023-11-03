import os
import pickle
from utils import check_and_overwrite
from typing import Optional

from utils import DocumentParser


# TODO: maybe implement a protocol here a corpus protocol


# manages individual corpus details
class Corpus:
    def __init__(self, directory_path: str):
        self.documents = []
        self.directory_path = directory_path
        self._load_documents()

    def _load_documents(self):
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




    # todo: implement different indexing granularity
    # todo: use appropriate index granularity based on corpus type
