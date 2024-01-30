import os
import pickle
from typing import List, Optional

from engine.VectorSpace import VectorSpace
from search_components import Document
from search_components.WordManager import CorpusWordManager


class Corpus:
    """Class representing the entire collection"""
    documents: List[Document]
    word_manager: CorpusWordManager
    vector_space: VectorSpace

    def __init__(self, directory_path: str):
        self.documents = []
        self.directory_path = directory_path
        self._load_documents()
        self.word_manager = CorpusWordManager(self.documents)
        self.vector_space = VectorSpace(self.word_manager)

    def _load_documents(self):
        from utils.Parser import DocumentParser
        parser = DocumentParser()
        # Check if directory exists
        if not os.path.exists(self.directory_path):
            raise FileNotFoundError(
                f"The directory {self.directory_path} does not exist"
            )
        # Iterate over each file in the directory
        for filename in os.listdir(self.directory_path):
            filepath = os.path.join(self.directory_path, filename)
            # Check if it's a file (and not a subdirectory)
            if os.path.isfile(filepath):
                doc = parser.parse(filepath)
                self.documents.append(doc)
        self.sort_corpus()

    def sort_corpus(self):
        self.documents.sort(
            key=lambda document: document.metadata.doc_id, reverse=False
        )

    def get_document_by_id(self, id: int) -> Optional[Document]:
        return self.documents[id]


class CorpusManager:
    """Class representing management of corpus we should only have one hence singleton"""
    _instance: Optional["CorpusManager"] = None
    raw_corpus = None

    def __init__(self) -> None:
        from utils.utilities import check_and_overwrite
        if os.path.exists("./pklfiles/CorpusManager.pkl"):
            file = open("./pklfiles/CorpusManager.pkl", "rb")
            self.raw_corpus = pickle.load(file).get_raw_corpus()

        else:
            self.raw_corpus = Corpus("./dataset/videogame")
            check_and_overwrite("./pklfiles/CorpusManager.pkl", self)


    @classmethod
    def get_instance(cls) -> "CorpusManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def get_raw_corpus(self) -> Corpus:
        return self.raw_corpus
