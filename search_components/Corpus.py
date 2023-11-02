from typing import List, Optional, Protocol


from search_components import Document
from utils import read_content



# TODO: maybe implement a protocol here a corpus protocol


# manages individual corpus details
class Corpus:
    def __init__(self, documents: List[Document]) -> None:
        self.document_list: List[Document] = documents


class CorpusFactory:
    def __init__(self, raw_corpus=Corpus(read_content())) -> None:
        self.corpus: Corpus = raw_corpus

    def get_raw_corpus(self) -> Corpus:
        return self.corpus



# singleton class to manage different types of corpus
class CorpusManager:
    _instance: Optional["CorpusManager"] = None

    def __init__(self) -> None:
        self.raw_corpus: Corpus = CorpusFactory().get_raw_corpus()
        self.tokanised_corpus: Corpus = self.raw_corpus
        self.lemmatised_corpus: Optional[Corpus] = None

    @classmethod
    def get_instance(cls) -> "CorpusManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def get_raw_corpus(self) -> Corpus:
        return self.raw_corpus

    # todo: implement different indexing granularity
    # todo: use appropriate index granularity based on corpus type
