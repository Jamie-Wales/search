from dataclasses import dataclass
from typing import List, Dict

from bs4 import BeautifulSoup


@dataclass
class DocumentMetaData:
    doc_id: int
    url: str
    esrb: str
    publisher: str
    genre: str
    developer: str


class Document:
    def __init__(self, raw_content: BeautifulSoup, text_content: str, metadata: DocumentMetaData,
                 tokenised_content: List[str]) -> None:
        self.raw_content = raw_content
        from engine import Vector

        self.text_content = text_content
        self.metadata = metadata
        self.tokenised_content: List[str] = tokenised_content
        self.document_frequency: Dict[str, int] = {}
        self._init_doc_frequency()
        self.vector = Vector()

    def _init_doc_frequency(self):
        for tokens in self.tokenised_content:
            count = self.document_frequency.get(tokens, 0)
            self.document_frequency[tokens] = count + 1
