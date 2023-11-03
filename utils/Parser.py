from typing import List, Any
from abc import ABC, abstractmethod
from bs4 import BeautifulSoup
from search_components import Document


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


class DocumentParser(IParser):
    def parse(self, path: str) -> Document:
        """
        Reads and parses a document from a given path.
        """
        raw_content, text_content = self._read_html(path)
        return Document(path=path, raw_content=raw_content, text_content=text_content)

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


class MetadataParser(IParser):
    def parse(self, path: str) -> List[DocumentMetaData]:
        """
        Reads and parses metadata from a given path.
        """
        # Your CSV reading and parsing logic goes here
        pass
