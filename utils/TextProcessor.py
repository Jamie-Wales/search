from typing import List

from nltk import word_tokenize


class DocumentProcessor:
    @staticmethod
    def tokenise(text: str) -> List[str]:
        """
        Tokenises the input text and returns a list of tokens.
        """
        return [token.casefold() for token in word_tokenize(text) if token.isalpha()]

    @staticmethod
    def stemm(text: str) -> List[str]:
        from nltk.stem import PorterStemmer
        ps = PorterStemmer()
        return [ps.stem(token) for token in word_tokenize(text) if token.isalpha()]

    @staticmethod
    def lemmatise(text: str) -> List[str]:
        """
        Lemmatises the input text and returns a list of lemmatised tokens.
        """
        from nltk.stem import WordNetLemmatizer
        lm = WordNetLemmatizer()
        return [lm.lemmatize(token) for token in word_tokenize(text) if token.text.isalpha()]
