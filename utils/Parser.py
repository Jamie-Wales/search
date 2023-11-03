import csv
from typing import List, Any, Optional
from abc import ABC, abstractmethod
from bs4 import BeautifulSoup
from search_components import Document, DocumentMetaData


class IParser(ABC):
    """
    Interface for a parser.
    """

    @abstractmethod
    def parse(self, data: Any) -> Any:
        """
        Parse the provided data and return the parsed result.

        Parameters:
        - data (Any): The data to parse.

        Returns:
        - Any: The parsed result.
        """
        pass


class MetadataParser(IParser):
    def __init__(self, csv_path: str = "./dataset/video-game-labels.csv"):
        self.metadata_dict = self.parse(csv_path)

    def parse(self, csv_path: str) -> dict:
        metadata_dict = dict()
        doc_id = 0
        with open(csv_path, "r+") as f:
            reader = csv.reader(f)
            for row in reader:
                doc_url, esrb, publisher, genre, developer = row
                clean_url = f"./dataset/videogame{doc_url.removeprefix("videogame/ps2.gamespy.com")}"
                meta = DocumentMetaData(doc_id, clean_url, esrb, publisher, genre, developer.strip())
                metadata_dict[clean_url] = meta
                doc_id += 1
        return metadata_dict

    def get_metadata_for_document(self, doc_url: str) -> DocumentMetaData:
        # Simply fetch and return the metadata using the URL as the key.
        if self.metadata_dict.get(doc_url) is None:
            print(doc_url)
        else:
            return self.metadata_dict.get(doc_url)


class DocumentParser(IParser):
    def __init__(self, metadata_parser: Optional[MetadataParser] = None):
        if metadata_parser is None:
            self.metadata_parser = MetadataParser()
        else:
            self.metadata_parser = metadata_parser

    def parse(self, path: str) -> Document:
        """
        Reads and parses a document from a given path.
        """
        raw_content, text_content = self._read_html(path)
        doc_metadata = self._read_metadata(self.metadata_parser, path)


        return Document(raw_content=raw_content, text_content=text_content, metadata=doc_metadata)

    @staticmethod
    def _read_metadata(metadata_parser: MetadataParser, path):
        return metadata_parser.get_metadata_for_document(path)

    @staticmethod
    def _read_html(path: str) -> (BeautifulSoup, str):
        """
        Reads the content from the given path and returns the raw and text content.
        """
        try:
            with open(path, "rb") as f:
                raw_content = BeautifulSoup(f.read(), features="html.parser")
                text_content = raw_content.getText(strip=True)
                return raw_content, text_content
        except FileNotFoundError:
            raise FileNotFoundError(f"The directory {path} does not exist")
        except PermissionError:
            raise PermissionError(f"Permission denied to access the directory {path}")
        except OSError as e:
            raise OSError(f"An OS error occurred: {e}")
