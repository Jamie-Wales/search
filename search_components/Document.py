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
            with open(path, "rb") as f:
                self.raw_content = BeautifulSoup(f.read(), features="html.parser")
                self.text_content = self.raw_content.getText(strip=True)
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
        output = []
        for token in token_doc:
            if token.text.isalpha():
                output.append(token)
        return output
