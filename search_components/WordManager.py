from typing import Dict

from search_components.Word import Word


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

    def get_count(self, word_type: str) -> int:
        if word_type not in ['original', 'stemmed', 'lemmatized']:
            raise ValueError("Invalid word type. Choose from 'original', 'stemmed', or 'lemmatized'.")
        count = sum(self.words_by_tag[word_type].values())
        return count

    def sort_words(self, word_type: str) -> None:
        if word_type not in self.words:
            raise ValueError("Invalid word type. Choose from 'original', 'stemmed', or 'lemmatized'.")

        self.words[word_type] = dict(sorted(self.words[word_type].items(), key=lambda item: item[0]))

    def from_document_managers(self, doc_managers_list):
        [self.add_word(word) for document in doc_managers_list for word in
         document.word_manager.words['original'].values()]
