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
        import pickle
        pickle.dump(obj, file)
        print(f"Data saved to '{string}.")


