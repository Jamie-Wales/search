from abc import ABC, abstractmethod
from typing import Optional, override

from nltk import PorterStemmer, WordNetLemmatizer


class IWord(ABC):
    def __init__(self, word: str, tag: str, stemmer: Optional[PorterStemmer] = None,
                 lemmer: Optional[WordNetLemmatizer] = None):
        self.original = word
        self.stemmed = None
        self.lemmatized = None
        self.tag = tag
        self.stem_word(word, stemmer)
        self.lemmatize_word(word, lemmer)

    @abstractmethod
    def stem_word(self, word: str, stemmer: Optional[PorterStemmer] = None) -> str:
        pass

    @abstractmethod
    def lemmatize_word(self, word: str, lemmer: Optional[WordNetLemmatizer] = None) -> str:
        pass


class Word(IWord):
    def __init__(self, word: str, tag: str, stemmer: Optional[PorterStemmer] = None,
                 lemmer: Optional[WordNetLemmatizer] = None):
        super().__init__(word, tag)
        self.original_concurrent = set()
        self.stemmed_concurrent = set()
        self.lemmatized_concurrent = set()

    @override
    def stem_word(self, word: str, stemmer: PorterStemmer = None):
        if stemmer is None:
            stemmer = PorterStemmer()
        self.stemmed = stemmer.stem(word)

    @override
    def lemmatize_word(self, word: str, lemmer: WordNetLemmatizer = None) -> str:
        if lemmer is None:
            lemmer = WordNetLemmatizer()
        self.lemmatized = lemmer.lemmatize(word)

    def get(self, type: str):
        return self.__getattribute__(f"{type}")

    def add_coccurrent(self, type: str, word: "Word") -> None:
        if type == "original":
            self.original_concurrent.add(word)
        elif type == "stemmed":
            self.stemmed_concurrent.add(word)
        elif type == "lemmatized":
            self.lemmatized_concurrent.add(word)
        else:
            raise ValueError("Invalid Type")


class QueryWord(Word):
    def __init__(self, word: str, stemmer: PorterStemmer = None, lemmer: WordNetLemmatizer = None):
        super().__init__(word, tag="query")


class NamedEntityWord(IWord):
    def __init__(self, word: str, type):
        super().__init__(word, tag="named entity")
        self.type = type


    @override
    def stem_word(self, word, stemmer: PorterStemmer = None) -> None:
        self.stemmed = word

    @override
    def lemmatize_word(self, word, lemmer: WordNetLemmatizer = None) -> None:
        self.lemmatized = word
