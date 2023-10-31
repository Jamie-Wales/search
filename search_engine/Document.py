from bs4 import BeautifulSoup
from typing import Union

class Document:
    def __init__(self, path: str) -> None:
        self.path: str = path
        self.raw_content: Union[BeautifulSoup, None] = None
        self.text_content: Union[str, None] = None
        try:
            with open(path, encoding='utf-8', errors="ignore") as f:
                self.raw_content = BeautifulSoup(f.read(), features="html.parser")
                self.text_content = self.raw_content.get_text()
        except FileNotFoundError:
            raise FileNotFoundError(f"The directory {self.path} does not exist")
        except PermissionError:
            raise PermissionError(f"Permission denied to access the directory {self.path}")
        except OSError as e:
            raise OSError(f"An OS error occurred: {e}")