import os
import pickle
import sys
from collections import defaultdict
from typing import List, Optional, Dict

from nltk import PorterStemmer, WordNetLemmatizer

from engine.Vector import QueryVector
from search_components import Document
from search_components.Word import QueryWord
from search_components.WordManager import QueryManager


class UserInput(object):
    _instance: Optional["UserInput"] = None

    def __init__(self):
        self._query = None

    def __new__(cls) -> "UserInput":
        if cls._instance is None:
            cls._instance = super(UserInput, cls).__new__(cls)
            cls._instance._query = ""
        return cls._instance

    def get_input(self, corpus_word_manager) -> QueryVector:
        return self.process_input(corpus_word_manager)

    def set_input(self) -> None:
        self._query = input("Please enter your search query: \n")
        if self._query == "d":
            sys.exit(0)

    def process_input(self, corpus_word_manager) -> QueryVector:
        from utils import DocumentProcessor
        dp = DocumentProcessor()
        tokens = dp.tokenise(self._query)
        stemmer = PorterStemmer()
        lemmar = WordNetLemmatizer()
        query_word_manager = QueryManager()
        for token in tokens:
            word = QueryWord(token, stemmer, lemmar)
            query_word_manager.add_word(word)

        vec = QueryVector(corpus_word_manager, query_word_manager)
        return vec


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
