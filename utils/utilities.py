import os
from collections import defaultdict
from typing import List, Optional, TypeVar, Dict

from search_components import Document

T = TypeVar('T')


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


def load(path: str):
    from pickle import load
    try:
        with open(path, "rb") as f:
            obj = load(f)
            f.close()
            print(f"Data saved to '{path}.")
            return obj
    except FileNotFoundError:
        print(f"The directory {path} does not exist")
        return None
    except PermissionError:
        raise PermissionError(f"Permission denied to access the directory {path}")
    except OSError as e:
        raise OSError(f"An OS error occurred: {e}")


def list_factory() -> list:
    return []


def inner_dict_factory() -> Dict[int, List[int]]:
    return defaultdict(list_factory)


def sort_documents(document_a: Document, document_b: Document):
    return document_a.metadata.doc_id < document_b.metadata.doc_id
