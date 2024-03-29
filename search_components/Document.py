from dataclasses import dataclass
from search_components import WordManager


@dataclass
class DocumentMetaData:
    """Metadata attributed to a document"""
    doc_id: int
    url: str
    esrb: str
    publisher: str
    genre: str
    developer: str


class Document:
    """Document in the collection"""
    def __init__(self, word_manager: WordManager, metadata: DocumentMetaData, raw_content):
        self.metadata = metadata
        self.word_manager = word_manager
        self.raw_content = raw_content
