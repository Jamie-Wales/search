import spacy
from typing import List


class TextProcessor:
    def __init__(self, model: str = "en_core_web_sm"):
        self.nlp = spacy.load(model)

    def tokenise(self, text: str) -> List[str]:
        """
        Tokenises the input text and returns a list of tokens.
        """
        token_doc = self.nlp(text)
        return [token.text for token in token_doc if token.text.isalpha()]

    def lemmatise(self, text: str) -> List[str]:
        """
        Lemmatises the input text and returns a list of lemmatised tokens.
        """
        token_doc = self.nlp(text)
        return [token.lemma_ for token in token_doc if token.text.isalpha()]

