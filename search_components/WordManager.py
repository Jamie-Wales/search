import heapq
from typing import Dict, List, Tuple, Type

from search_components import Word
from search_components.Word import NamedEntityWord


class WordManager:

    def __init__(self):
        self.words_by_tag: Dict[str, Dict[str, Dict[str, int]]] = {}
        self.words: Dict[str, Dict[str, Type[Word]]] = {
            'original': {},
            'stemmed': {},
            'lemmatized': {}
        }

    def add_word(self, word: Type[Word]) -> None:
        word_types = [('original', word.original), ('stemmed', word.stemmed), ('lemmatized', word.lemmatized)]

        for word_type, word_form in word_types:
            if word_form not in self.words[word_type]:
                self.words[word_type][word_form] = word
            if isinstance(word, NamedEntityWord):
                self.words_by_tag.setdefault(word_type, {}).setdefault(word_form, {}).setdefault(word.tag, 0)
                self.words_by_tag[word_type][word_form][word.tag] += 1
            else:
                for tag in word.tag:
                    self.words_by_tag.setdefault(word_type, {}).setdefault(word_form, {}).setdefault(tag, 0)
                    self.words_by_tag[word_type][word_form][tag] += 1

    def get_tag_and_count(self, word_type: str, word: str) -> List[Tuple[str, int]]:
        """
        Returns a list of tuples containing (tag, count) for the specified word_type and word.
        """
        if word_type not in ['original', 'stemmed', 'lemmatized']:
            raise ValueError("Invalid word type. Choose from 'original', 'stemmed', or 'lemmatized'.")

        tag_counts = self.words_by_tag.get(word_type, {}).get(word, {})
        return list(tag_counts.items())

    def sort_words(self, word_type: str) -> None:
        if word_type not in self.words:
            raise ValueError("Invalid word type. Choose from 'original', 'stemmed', or 'lemmatized'.")

        self.words[word_type] = dict(sorted(self.words[word_type].items(), key=lambda item: item[0]))

    def get_word_count(self, word_type: str, word_form: str) -> int:
        total = 0
        for count in self.words_by_tag.get(word_type).get(word_form).values():
            total += count

        return total

    def get_word(self, type, word_form: str) -> Type[Word]:
        if type not in self.words:
            ValueError("Invalid type")
        else:
            return self.words[type][word_form]


class CorpusWordManager(WordManager):
    def __init__(self, document_list):
        super().__init__()
        self.count = {"original": {}, "stemmed": {}, "lemmatized": {}}
        self.number_of_documents = 0
        self.from_document_managers(document_list)
        self.word_matrix = None

    def from_document_managers(self, doc_managers_list):
        for document in doc_managers_list:
            self.number_of_documents += 1
            seen = {"original": set(), "stemmed": set(), "lemmatized": set()}

            for word in document.word_manager.words["original"].values():
                self.add_word(word)
                # Mark the word as seen in this document
                seen["original"].add(word.original)
                seen["stemmed"].add(word.stemmed)
                seen["lemmatized"].add(word.lemmatized)

            # Update document counts for each word type
            for word_type in ["original", "stemmed", "lemmatized"]:
                for word in seen[word_type]:
                    self.count[word_type][word] = self.count[word_type].get(word, 0) + 1

    def generate_word_matrix(self, matrix):
        self.word_matrix = matrix

    def generate_concurrent_words(self):
        lemmatized_keys = list(self.word_matrix["lemmatized"].keys())

        for i, check_word in enumerate(lemmatized_keys):
            current_word = self.get_word("lemmatized", check_word)
            heap = []

            # Start the inner loop from the key ahead of the current key
            for j in range(i + 1, len(lemmatized_keys)):
                word = lemmatized_keys[j]

                co_correntdocs = set(self.word_matrix["lemmatized"][check_word]).union(
                    self.word_matrix["lemmatized"][word])

                sum = 0  # Reset sum for each word
                for co_occurring in co_correntdocs:
                    original_word = self.word_matrix['lemmatized'][check_word].get(co_occurring, 0)
                    compare_word = self.word_matrix['lemmatized'][word].get(co_occurring, 0)
                    sum += original_word * compare_word

                if sum > 0:
                    heapq.heappush(heap, (sum, word))

                while len(heap) > 2:
                    heapq.heappop(heap)

            while len(heap) != 0:
                current_word.add_coccurrent("lemmatized", heapq.heappop(heap)[1])


class QueryManager:
    def __init__(self):
        self.words: Dict[str, Dict[str, int]] = {
            'original': {},
            'stemmed': {},
            'lemmatized': {}
        }

    def add_word(self, word: Type[Word]) -> None:
        word_types = [('original', word.original), ('stemmed', word.stemmed), ('lemmatized', word.lemmatized)]

        for word_type, word_form in word_types:
            # Ensure the word_type key exists
            if word_type not in self.words:
                self.words[word_type] = {}

            # Ensure the word_form key exists
            if word_form not in self.words[word_type]:
                self.words[word_type][word_form] = 0

            # Increment the count
            self.words[word_type][word_form] += 1

    def get_word_count(self, word_type: str, word: str) -> int:
        return self.words.get(word_type).get(word, 0)
