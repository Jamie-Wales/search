import csv
from abc import ABC, abstractmethod
from typing import Any, Optional

from bs4 import BeautifulSoup, Comment

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

    def parse(self, path: str, okm25f=True) -> Document:
        """
        Reads and parses a document from a given path.
        """
        from utils import DocumentProcessor
        doc_processor = DocumentProcessor()
        raw_content, structured_content = self._read_html(path, okm25f)
        doc_metadata = self._read_metadata(self.metadata_parser, path)

        # Initialize a dictionary to hold word counts per tag type
        tag_word = {}
        tokenised_content = []

        # Process each tag type, tokenize the text and count the words
        for content in structured_content:
            tokens = doc_processor.tokenise(content[0])
            for token in tokens:
                tokenised_content.append(token)
                tag_word[content[1]] = tokens
        return Document(raw_content, structured_content, doc_metadata, tokenised_content, tag_word)

    @staticmethod
    def _read_metadata(metadata_parser: MetadataParser, path):
        return metadata_parser.get_metadata_for_document(path)

    @staticmethod
    def _read_html(path: str, okm25f=True) -> (BeautifulSoup, str):
        """
        Reads the content from the given path and returns the raw and text content.
        """
        try:
            with open(path, "rb") as f:
                from utils import DocumentProcessor
                dp = DocumentProcessor()
                raw_content = BeautifulSoup(f.read(), features="html.parser")
                for ele in raw_content(["script", "img", "style", "a"]):
                    ele.extract()


                for comment in raw_content.find_all(string=lambda text: isinstance(text, Comment)):
                    comment.extract()

                for ele in raw_content.select("#footer"):
                    ele.extract()

                for ele in raw_content.select("#menuLeft"):
                    ele.extract()

                for ele in raw_content.select("#headerSearch"):
                    ele.extract()


                text_content = ""
                output = []

                if okm25f is False:
                    for ele in raw_content.select("#content") + " " + raw_content.find_all("title"):
                        text_content += ele.getText() + " "
                    return raw_content, text_content.strip()

                if okm25f is True:
                    for ele in raw_content.find_all():
                        output.append((ele.getText(), ele.name))
                    return raw_content, output

        except FileNotFoundError:
            raise FileNotFoundError(f"The directory {path} does not exist")
        except PermissionError:
            raise PermissionError(f"Permission denied to access the directory {path}")
        except OSError as e:
            raise OSError(f"An OS error occurred: {e}")
