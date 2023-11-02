from typing import Union, List

from bs4 import BeautifulSoup
import spacy


class Document:
    def __init__(self, path: str) -> None:
        self.path: str = path
        self.raw_content: Union[BeautifulSoup, None] = None
        self.text_content: Union[str, None] = None
        self.tokenised_content: Union[List[str], None] = None
        try:
            with open(path, "r+") as f:
                self.raw_content = BeautifulSoup(f.read(), features="html.parser")
                self.text_content = self.raw_content.get_text()
                self.tokenised_content = self.tokenise(self.text_content)
        except FileNotFoundError:
            raise FileNotFoundError(f"The directory {self.path} does not exist")
        except PermissionError:
            raise PermissionError(f"Permission denied to access the directory {self.path}")
        except OSError as e:
            raise OSError(f"An OS error occurred: {e}")

    @staticmethod
    def tokenise(document):
        nlp = spacy.load("en_core_web_sm")
        token_doc = nlp(document)
        return [token.text for token in token_doc]