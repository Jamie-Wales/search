from search_engine import Document
from utils import read_content
from typing import List, Optional


# Manages individual corpus details
class Corpus:
    document_list: List[Optional["Document"]] = []

    def generate_raw_corpus(self) -> None:
        self.document_list = read_content()


# Singleton class to manage different types of corpus
class CorpusManager(object):
    _instance = None  # Singleton instance variable

    def __new__(cls) -> "CorpusManager":
        if cls._instance is None:
            cls._instance = super(CorpusManager, cls).__new__(cls)
            cls._instance.raw_corpus = Corpus()
            cls._instance.raw_corpus.generate_raw_corpus()
            cls._instance.lemmatised_corpus = None
            cls._instance.tokenized_corpus = None
        return cls._instance

    # TODO: Implement different indexing granularity
    # TODO: Use appropriate index granularity based on corpus type
