from engine import DocumentVectorStore
from engine import Ranker
from search_components import CorpusManager
from utils import UserInput, SpellChecker


class Search:
    def __init__(self):
        self.document_vector_store = DocumentVectorStore()
        self.corpus_manager = CorpusManager().get_raw_corpus()
        if self.document_vector_store.need_vector_generation:
            self.document_vector_store.generate_vectors(self.corpus_manager)
        self.input = UserInput()
        self.spellVec = None

    def search(self, vec_type: str, usr_input: str):
        self.input.set_input(usr_input)
        user_input = self.input.get_input(self.corpus_manager.word_manager, self.corpus_manager.vector_space)
        spellCheck = SpellChecker(user_input.word_manager, self.corpus_manager.word_manager, self.corpus_manager
                                  .vector_space)
        spellCheck.correct_words()
        return Ranker().tf_idf_vector(vec_type, self.document_vector_store, user_input), spellCheck

