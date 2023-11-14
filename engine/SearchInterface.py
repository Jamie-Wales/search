from engine import Engine
from utils import UserInput, load, check_and_overwrite


class SearchInterface:
    engine = None

    def __init__(self):
        self.engine = load("Engine.pkl")
        if self.engine is None:
            self.engine = Engine()
            check_and_overwrite("Engine.pkl", self.engine)
        self.input = UserInput()

    def search(self):
        while self.input.continue_input():
            self.input.set_input()
            raw_input = self.input.get_input()
            from utils import SpellChecker
            processed_input = self.input.process_input(raw_input)
            sp = SpellChecker(processed_input, self.engine.corpus.term_frequency)
            ranking = self.engine.process_query(sp.correct_words())
            self.print_ranking(ranking)

    def print_ranking(self, ranked_documents):
        print("Top 10 Ranked Documents:")
        print("-" * 80)
        # Add 'Score' to the format string
        format_string = "{:<10} {:<45} {:<10} {:<15} {:<20} {:<20} {:<10}"
        print(format_string.format("Rank", "URL", "ESRB", "Publisher", "Genre", "Developer", "Score"))

        for rank, (metadata, score) in enumerate(ranked_documents[:10], 1):
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
