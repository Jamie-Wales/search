from __future__ import annotations

import os
import pickle

from search_components.Word import QueryWord
from search_components.WordManager import QueryManager


class UserInput:
    """Class that handles turning input into query vector"""

    @staticmethod
    def process_input(input, corpus_word_manager, vector_space, stemmar, lemmar) -> "QueryVector":
        """Tokenises the input and turns into vector representation"""
        from utils.TextProcessor import DocumentProcessor
        dp = DocumentProcessor()
        tokens = dp.tokenise(input)
        query_word_manager = QueryManager()
        for token in tokens:
            word = QueryWord(token, stemmar, lemmar)
            query_word_manager.add_word(word)

        from vec.Vector import QueryVector
        vec = QueryVector(corpus_word_manager, query_word_manager, "query", vector_space)
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
