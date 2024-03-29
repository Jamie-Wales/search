import string
from typing import List

from nltk import word_tokenize


class DocumentProcessor:
    """Class for various textual processing methods"""
    @staticmethod
    def remove_punctuation(text: str) -> str:
        """Remove punctuation - are spaces due to urls and all other punc is '' """
        text = text.replace('-', ' ')
        translator = str.maketrans('', '', string.punctuation)
        return text.translate(translator)

    @staticmethod
    def tokenise(text: str) -> List[str]:
        """
        Tokenises the input text and returns a list of tokens.
        """
        text_no_punc = DocumentProcessor.remove_punctuation(text)
        lowercase_tokens = [token.casefold() for token in word_tokenize(text_no_punc) if text_no_punc is not None]
        return lowercase_tokens
