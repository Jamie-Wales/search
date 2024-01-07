from __future__ import annotations
import os
import pickle
from collections import defaultdict
from typing import List, Optional, Dict

from search_components import Document
from search_components.Word import QueryWord
from search_components.WordManager import QueryManager


class UserInput:

    @staticmethod
    def process_input(input, corpus_word_manager, vector_space, stemmar, lemmar) -> "QueryVector":
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


def list_factory() -> list:
    return []


def inner_dict_factory() -> Dict[int, List[int]]:
    return defaultdict(list_factory)


def sort_documents(document_a: Document, document_b: Document):
    return document_a.metadata.doc_id < document_b.metadata.doc_id
