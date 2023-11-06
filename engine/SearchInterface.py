from engine import Engine
from utils import UserInput


class SearchInterface:
    def __init__(self):
        self.engine = Engine()
        self.input = UserInput()

    def search(self):
        while self.input.continue_input():
            self.input.set_input()
            input = self.input.get_input()
            input = self.input.process_input(input)

            # TO DO: remove punkt
            self.print_ranking(self.engine.update_ranking(input))

    def print_ranking(self, ranked_documents):
        print("Top 20 Ranked Documents:")
        print("-" * 80)
        # Add 'Score' to the format string
        format_string = "{:<10} {:<40} {:<10} {:<15} {:<20} {:<20} {:<10}"
        print(format_string.format("Rank", "URL", "ESRB", "Publisher", "Genre", "Developer", "Score"))

        for rank, (metadata, score) in enumerate(ranked_documents[:20], 1):
            print(format_string.format(
                str(rank),
                metadata.url[:37] + '...' if len(metadata.url) > 40 else metadata.url,
                metadata.esrb,
                metadata.publisher[:12] + '...' if len(metadata.publisher) > 15 else metadata.publisher,
                metadata.genre[:17] + '...' if len(metadata.genre) > 20 else metadata.genre,
                metadata.developer[:17] + '...' if len(metadata.developer) > 20 else metadata.developer,
                f"{score:.5f}"  # Format the score to show 5 decimal places
            ))

        print("-" * 80)
