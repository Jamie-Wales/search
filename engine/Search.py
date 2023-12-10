from engine.DocumentVectorStore import DocumentVectorStore
from engine.Ranker import Ranker
from search_components import CorpusManager
from utils import UserInput


class Search:
    def __init__(self):
        self.document_vector_store = DocumentVectorStore()
        if self.document_vector_store.need_vector_generation:
            self.document_vector_store.generate_vectors(CorpusManager().get_raw_corpus())
        self.input = UserInput()

    def search(self):
        while True:
            self.input.set_input()
            rank = Ranker()
            rank.tf_idf_vector(self.document_vector_store,
                               self.input.get_input(CorpusManager().get_raw_corpus().word_manager))

            print(rank.tf_idf_documents("lemmatized"))