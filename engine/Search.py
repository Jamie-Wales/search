from engine import DocumentVectorStore
from engine import Ranker
from search_components import CorpusManager
from utils import UserInput


class Search:
    def __init__(self):
        self.document_vector_store = DocumentVectorStore()
        self.corpus_manager = CorpusManager().get_raw_corpus()
        if self.document_vector_store.need_vector_generation:
            self.document_vector_store.generate_vectors(self.corpus_manager)
        self.input = UserInput()

    def search(self, input: str):
            self.input.set_input(input)
            rank = Ranker()
            rank.tf_idf_vector(self.document_vector_store,
                               self.input.get_input(self.corpus_manager.word_manager, self.corpus_manager.vector_space))
            rank.tf_idf_documents("lemmatized")
            return rank

    def print_ranking(self, ranked_documents):
        print("Top 10 Ranked Documents:")
        print("-" * 80)
        # Add 'Score' to the format string
        format_string = "{:<10} {:<45} {:<10} {:<15} {:<20} {:<20} {:<10}"
        print(format_string.format("Rank", "URL", "ESRB", "Publisher", "Genre", "Developer", "Score"))

        for rank, (score, doc_id) in enumerate(ranked_documents[:10], 1):
            metadata = CorpusManager().get_raw_corpus().get_document_by_id(doc_id).metadata
            print(format_string.format(
                str(rank),
                metadata.url[:39] + '...' if len(metadata.url) > 45 else metadata.url,
                metadata.esrb,
                metadata.publisher[:12] + '...' if len(metadata.publisher) > 15 else metadata.publisher,
                metadata.genre[:17] + '...' if len(metadata.genre) > 20 else metadata.genre,
                metadata.developer[:17] + '...' if len(metadata.developer) > 20 else metadata.developer,
                f"{score:.5f}"  # Format the score to show 5 decimal places
            ))

        print("-" * 80)
