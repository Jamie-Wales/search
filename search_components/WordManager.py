from typing import Dict, List, Tuple

from search_components.Word import Word, QueryWord


class WordManager:
    def __init__(self):
        self.words_by_tag: Dict[str, Dict[str, Dict[str, int]]] = {}
        self.words: Dict[str, Dict[str, Word]] = {
            'original': {},
            'stemmed': {},
            'lemmatized': {}
        }

    def add_word(self, word: Word) -> None:
        word_types = [('original', word.original), ('stemmed', word.stemmed), ('lemmatized', word.lemmatized)]

        for word_type, word_form in word_types:
            if word_form not in self.words[word_type]:
                self.words[word_type][word_form] = word
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
        for count in self.words_by_tag.get(word_type).get(word_form, {"": 0}).values():
            total += count

        return total


class CorpusWordManager(WordManager):
    def __init__(self):
        super().__init__()
        self.n_doc_count = {"original": {}, "stemmed": {}, "lemmatized": {}}
        self.number_of_documents = 0

    def from_document_managers(self, doc_managers_list):
        for document in doc_managers_list:
            self.number_of_documents += 1
            for word in document.word_manager.words['original'].values():
                self.add_word(word)
                self.n_doc_count["lemmatized"].setdefault(word.lemmatized, 0)
                self.n_doc_count["lemmatized"][word.lemmatized] += 1
                self.n_doc_count["stemmed"].setdefault(word.stemmed, 0)
                self.n_doc_count["stemmed"][word.stemmed] += 1
                self.n_doc_count["original"].setdefault(word.original, 0)
                self.n_doc_count["original"][word.original] += 1


class QueryManager:
    def __init__(self):
        self.words: Dict[str, Dict[str, int]] = {
            'original': {},
            'stemmed': {},
            'lemmatized': {}
        }

    def add_word(self, word: QueryWord) -> None:
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

    def get_word_count(self, word_type: str, word_form: str) -> int:
        return self.words.get(word_type).get(word_form, 0)


