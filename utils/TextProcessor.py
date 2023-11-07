import string
from typing import List

from nltk import word_tokenize


class DocumentProcessor:
    @staticmethod
    def tokenise(text: str, stem=False, lem=True) -> List[str]:
        """
        Tokenises the input text and returns a list of tokens.
        """

        lowercase_tokens = []
        if not stem and not lem:
            lowercase_tokens = [token.casefold() for token in word_tokenize(text) if token.isalpha()]
        if stem:
            lowercase_tokens = DocumentProcessor.stemm(text)
        elif lem:
            lowercase_tokens = DocumentProcessor.lemmatise(text)
        return lowercase_tokens

    @staticmethod
    def stemm(text: str) -> List[str]:
        from nltk.stem import PorterStemmer
        ps = PorterStemmer()
        output = []
        for token in word_tokenize(text):
            output.append(ps.stem(token))
        return output

    @staticmethod
    def lemmatise(text: str) -> List[str]:
        """
        Lemmatises the input text and returns a list of lemmatised tokens.
        """
        from nltk.stem import WordNetLemmatizer
        lm = WordNetLemmatizer()
        output = []
        for token in word_tokenize(text):
            if token.isalpha():
                output.append(lm.lemmatize(token.casefold()))
        return output

    @staticmethod
    def punct_remove(text: str):
        return text.replace(string.punctuation, " ")
