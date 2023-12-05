import os
import pickle
from collections import defaultdict
from typing import List, Optional, Dict

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

    def set_input(self) -> None:
        self._query = input("Please enter your search query: \n")

    def continue_input(self) -> bool:
        return self._query != "d"

    def process_input(self, string_input):
        from utils import DocumentProcessor
        dp = DocumentProcessor()
        return dp.tokenise(string_input)


def check_and_overwrite(string: str, obj: object):
    """
    Checks if a file exists, and prompts the user whether to overwrite it or not.
    If the user chooses to overwrite or the file doesn't exist, it writes the data to the file.
    """
    if os.path.exists(string):
        response = input(f"'{string}' already exists. Do you want to overwrite it? (yes/no): ").lower()

        if response != 'yes':
            print("Operation aborted by user.")
            return

    with open(string, 'wb') as file:
        from pickle import dump
        dump(obj, file)
        print(f"Data saved to '{string}.")


def load(url):
    try:
        with open(url, 'rb') as f:
            return pickle.load(f)
    except (FileNotFoundError, EOFError, pickle.UnpicklingError):
        return None


def list_factory() -> list:
    return []


def inner_dict_factory() -> Dict[int, List[int]]:
    return defaultdict(list_factory)


def sort_documents(document_a: Document, document_b: Document):
    return document_a.metadata.doc_id < document_b.metadata.doc_id
