from dataclasses import dataclass
from typing import List, Dict

from bs4 import BeautifulSoup


@dataclass
class DocumentMetaData:
    doc_id: int
    url: str  # Changed from 'path' to 'url'
    esrb: str
    publisher: str
    genre: str
    developer: str


class Document:
    def __init__(self, raw_content: BeautifulSoup, text_content: str, metadata: DocumentMetaData,
                 tokenised_content: List[str])  -> None:
        self.raw_content = raw_content
        self.text_content = text_content
        self.metadata = metadata
        self.tokenised_content: List[str] = tokenised_content
        self.document_frequency: Dict[str, int] = {}
        self.init_document_freq()


    def init_document_freq(self):
        for tokens in self.tokenised_content:
            self.document_frequency.setdefault(tokens, 0)
            self.document_frequency[tokens] = self.document_frequency[tokens] + 1
