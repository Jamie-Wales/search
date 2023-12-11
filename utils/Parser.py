import csv
from abc import ABC, abstractmethod
from typing import Any, Optional

from bs4 import BeautifulSoup, Comment
from nltk import PorterStemmer
from nltk import WordNetLemmatizer

from search_components import Document, DocumentMetaData
from search_components.Word import Word
from search_components.WordManager import WordManager


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
        content = self._read_html(path)
        doc_metadata = self._read_metadata(self.metadata_parser, path)
        word_manager = WordManager()
        stemmer = PorterStemmer()
        lemmar = WordNetLemmatizer()
        # Process each tag type, tokenize the text and count the word
        for element in content:
            tokens = doc_processor.tokenise(element[0])
            for count, token in enumerate(tokens):
                word = Word(token, element[1], count, stemmer, lemmar)
                word_manager.add_word(word)
        return Document(word_manager, doc_metadata)

    @staticmethod
    def _read_metadata(metadata_parser: MetadataParser, path):
        return metadata_parser.get_metadata_for_document(path)

    @staticmethod
    def get_element_texts(element):
        output = []
        # Process meta tags
        meta_tags = element.find_all("meta")
        for ele in meta_tags:
            if ele is not None:
                content = ele.get('content')  # Safer access to 'content' attribute
                output.append((content, ["meta"]))

        # Process div elements
        divs = element.find("div", id="content")
        for child in divs.findChildren():
            if child is not None:
                class_attr = child.get('class')  # Safer access to 'class' attribute
                if class_attr is not None:
                    output.append((child.get_text(), class_attr))
                else:
                    output.append((child.get_text(), ["div"]))
            else:
                output.append((divs.get_text(),  ["div"]))

        return output

    @staticmethod
    def _read_html(path: str):
        """
        Reads the content from the given path and returns the raw and text content.
        """
        try:
            with open(path, "rb") as f:
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

                output = DocumentParser.get_element_texts(raw_content)
                return output

        except FileNotFoundError:
            raise FileNotFoundError(f"The directory {path} does not exist")
        except PermissionError:
            raise PermissionError(f"Permission denied to access the directory {path}")
        except OSError as e:
            raise OSError(f"An OS error occurred: {e}")
