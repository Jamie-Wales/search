from nltk import WordNetLemmatizer, PorterStemmer

from engine.Ranker import Ranker
from search_components.NamedEntityRecogniser import NamedEntityRecogniser
from utils.utilities import UserInput
from search_components.Corpus import CorpusManager
from vec.DocumentVectorStore import DocumentVectorStore

class Search:
    def __init__(self):
        self.document_vector_store = DocumentVectorStore()
        self.corpus_manager = CorpusManager()
        if self.document_vector_store.need_vector_generation:
            self.document_vector_store.generate_vectors(self.corpus_manager.get_raw_corpus())
        self.spellVec = None
        self.lemmar = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        self.search_input = None
        self.named_entities = NamedEntityRecogniser(self.corpus_manager.get_raw_corpus().documents)
        print(self.named_entities.find_words_with_prefix("re"))
    def search(self, vec_type: str, usr_input: str, name_entities) -> list:
        self.search_input = None
        self.search_input = UserInput.process_input(usr_input, self.corpus_manager.get_raw_corpus().word_manager,
                                                    self.corpus_manager.get_raw_corpus().vector_space, self.stemmer,
                                                    self.lemmar)
        self.search_input.query_expansion(vec_type)
        return Ranker.tf_idf_vector(vec_type, self.document_vector_store, self.search_input, name_entities)

    def rerank(self, vec_type: str, usr_input: "QueryVector") -> list:
        self.search_input = usr_input
        return Ranker.tf_idf_vector(vec_type, self.document_vector_store, usr_input, {})

    def relevance_feedback(self, relevant_document_id: list[int]):
        relevant_docs = []

        if len(relevant_document_id) == 0:
            return
        else:
            for id in relevant_document_id:
                relevant_docs.append(self.document_vector_store.get_vector(id).TFIDFVector)
                return self.search_input.relevance_feedback("lemmatized", relevant_docs)
