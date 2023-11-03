import spacy
from typing import List
from search_components import Document, Corpus


class DocumentProcessor:
    def __init__(self, model: str = "en_core_web_sm"):
        self.nlp = spacy.load(model)

    def tokenise(self, text: Document) -> List[str]:
        """
        Tokenises the input text and returns a list of tokens.
        """
        token_doc = self.nlp(text.text_content)
        return [token.text for token in token_doc if token.text.isalpha()]

    def lemmatise(self, text: Document) -> List[str]:
        """
        Lemmatises the input text and returns a list of lemmatised tokens.
        """
        token_doc = self.nlp(text.text_content)
        return [token.lemma_ for token in token_doc if token.text.isalpha()]

    def processCorpus(self, corpus: Corpus):
        for document in corpus.documents:
            document.tokenised_content = self.tokenise(document.text_content)




