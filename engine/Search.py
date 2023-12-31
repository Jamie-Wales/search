from nltk import WordNetLemmatizer, PorterStemmer
from engine import DocumentVectorStore
from engine import Ranker
from engine.Vector import QueryVector
from search_components import CorpusManager
from utils import UserInput


class Search:
    def __init__(self):
        self.document_vector_store = DocumentVectorStore()
        self.corpus_manager = CorpusManager().get_raw_corpus()
        if self.document_vector_store.need_vector_generation:
            self.document_vector_store.generate_vectors(self.corpus_manager)
        self.spellVec = None
        self.lemmar = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        self.search_input = None

    def search(self, vec_type: str, usr_input: str) -> list:
        self.search_input = None
        self.search_input = UserInput.process_input(usr_input, self.corpus_manager.word_manager, self.corpus_manager.vector_space, self.stemmer, self.lemmar)
        return Ranker.tf_idf_vector(vec_type, self.document_vector_store, self.search_input)

    def rerank(self, vec_type: str, usr_input: QueryVector) -> list:
        self.search_input = None
        return Ranker.tf_idf_vector(vec_type, self.document_vector_store, usr_input)
