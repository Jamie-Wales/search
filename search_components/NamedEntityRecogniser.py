from typing import List

import spacy
from pygtrie import CharTrie

from search_components import Document
from search_components.Word import NamedEntityWord
from utils.utilities import check_and_overwrite, load


class NamedEntityRecogniser:
    tree = None

    def __init__(self, list_of_documents: List[Document]):
        self.tree = load("./ner.pkl")
        if self.tree is None:
            self.tree = CharTrie()
            nlp = spacy.load('en_core_web_sm')
            for docs in list_of_documents:
                doc = nlp(docs.raw_content)
                for entity in doc.ents:
                    self.tree[f"{entity.text.lower()}"] = (entity.text, entity.label_)
                    docs.word_manager.add_word(NamedEntityWord(f"{entity.text}, {entity.label_}", entity.label_))
            check_and_overwrite("./ner.pkl", self.tree)

    def find_words_with_prefix(self, prefix):
        out = []
        try:
            out = list(self.tree.itervalues(prefix=prefix))
        finally:
            return out
