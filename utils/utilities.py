import os
from typing import List, Optional

from search_components import Document


class UserInput(object):
    _instance: Optional["UserInput"] = None

    def __init__(self):
        self._query = None

    def __new__(cls) -> "UserInput":
        if cls._instance is None:
            cls._instance = super(UserInput, cls).__new__(cls)
            cls._instance._query = ""
        return cls._instance

    def get_input(self) -> Optional[str]:
        return self._query

    def set_input(self, user_input: str) -> None:
        self._query = user_input

    def continue_input(self) -> bool:
        return self._query != "d"


def read_content(path: str = "./dataset/videogames/") -> List[Document]:
    output_list: List[Document] = []

    # Loop through all files in the directory
    for file_path in os.listdir(path):
        full_path = os.path.join(path, file_path)

        # Create Document objects and add them to the list
        doc = Document(full_path)
        output_list.append(doc)

    return output_list
