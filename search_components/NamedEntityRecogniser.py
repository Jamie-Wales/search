import spacy
from pygtrie import CharTrie

from search_components.Corpus import Corpus
from search_components.Word import NamedEntityWord
from utils.utilities import check_and_overwrite, load


class NamedEntityRecogniser:
    tree = None

    def __init__(self, corpus: Corpus) -> None:
        self.tree = load("./pklfiles/ner.pkl")
        if self.tree is None:
            self.tree = CharTrie()
            nlp = spacy.load('en_core_web_sm')
            for docs in corpus.documents:
                doc = nlp(docs.raw_content)
                for entity in doc.ents:
                    self.tree[f"{entity.text.lower()}"] = (entity.text, entity.label_)
                    ner_word = NamedEntityWord(f"{entity.text}, {entity.label_}", entity.label_)
                    docs.word_manager.add_word(ner_word)
                    corpus.word_manager.add_word(ner_word)
            check_and_overwrite("./pklfiles/ner.pkl", self.tree)

    def find_words_with_prefix(self, prefix):
        out = []
        try:
            out = list(self.tree.itervalues(prefix=prefix))
        finally:
            return out
