from dataclasses import dataclass
from typing import Union, List

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
    def __init__(self, raw_content: BeautifulSoup, text_content: str, metadata: DocumentMetaData) -> None:
        self.raw_content = raw_content
        self.text_content = text_content
        self.metadata = metadata
        self.tokenised_content: Union[List[str], None] = None
