import string
from typing import List

from nltk import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer


class DocumentProcessor:
    @staticmethod
    def remove_punctuation(text: str) -> str:
        text = text.replace('-', ' ')
        translator = str.maketrans('', '', string.punctuation)
        return text.translate(translator)

    @staticmethod
    def tokenise(text: str, stem=False, lem=True) -> List[str]:
        """
        Tokenises the input text and returns a list of tokens.
        """
        text_no_punc = DocumentProcessor.remove_punctuation(text)

        lowercase_tokens = [token.casefold() for token in word_tokenize(text_no_punc) if text_no_punc is not None]

        if stem:
            return DocumentProcessor.stemm(lowercase_tokens)
        elif lem:
            return DocumentProcessor.lemmatise(lowercase_tokens)

        return lowercase_tokens

    @staticmethod
    def stemm(tokens: List[str]) -> List[str]:
        ps = PorterStemmer()
        return [ps.stem(token) for token in tokens]

    @staticmethod
    def lemmatise(tokens: List[str]) -> List[str]:
        lm = WordNetLemmatizer()
        return [lm.lemmatize(token) for token in tokens]
