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

