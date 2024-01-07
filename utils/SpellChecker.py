import numpy as np

from engine.VectorSpace import VectorSpace
from search_components.Word import Word, QueryWord
from search_components.WordManager import QueryManager, CorpusWordManager


class SpellChecker:
    def __init__(self, corpus_word_manager: CorpusWordManager, vector_space: VectorSpace):
        self.corpus_word_manager = corpus_word_manager
        self.vector_space = vector_space
        self.corrected_words = {}
        self.corrected_vector = None

    def correct_words(self, input: QueryManager) -> None:
        self.corrected_words.clear()
        self.corrected_vector = None
        for word in input.words["original"].keys():
            if word not in self.corpus_word_manager.words["original"].keys():
                closest_word, min_distance = self.find_closest_word_in_wm(word)
                if min_distance < 3:
                    print(f"changing {word} to {closest_word.original}")
                    self.corrected_words[word] = closest_word
        new_word_manager = QueryManager()
        for word in input.words["original"].keys():
            if word in self.corrected_words:
                new_word_manager.add_word(self.corrected_words.get(word))
            else:
                new_word_manager.add_word(QueryWord(word))

        if len(self.corrected_words) > 0:
            from vec.Vector import QueryVector
            self.corrected_vector = QueryVector(self.corpus_word_manager, new_word_manager, "", self.vector_space)

    def find_closest_word_in_wm(self, input_word: str) -> tuple[Word | str, float | int]:
        min_distance = float('inf')
        closest_word = input_word
        for word in self.corpus_word_manager.words["original"].values():
            if abs(len(input_word) - len(word.original)) > 2:
                # Skip edit distance calculation if length difference is more than 2
                continue

            distance = self.edit_distance(input_word, word.original)
            if distance < min_distance:
                min_distance = distance
                closest_word = word
        return closest_word, min_distance

    @staticmethod
    def edit_distance(s1: str, s2: str) -> int:
        s1 = ' ' + s1
        s2 = ' ' + s2

        m = np.zeros((len(s1), len(s2)))

        for i in range(len(s1)):
            m[i][0] = i
        for j in range(len(s2)):
            m[0][j] = j

        for i in range(1, len(s1)):
            min_value_in_row = float('inf')
            for j in range(1, len(s2)):
                offset = 0 if s1[i] == s2[j] else 1
                m[i][j] = min(m[i - 1][j - 1] + offset, m[i - 1][j] + 1, m[i][j - 1] + 1)
                min_value_in_row = min(min_value_in_row, m[i][j])

            # Early exit if the edit distance is already above 2
            if min_value_in_row > 2:
                return 3  # Return 3 indicating edit distance > 2

        # Check the last row's minimum value
        if min(m[-1]) > 2:
            return 3  # Return 3 indicating edit distance > 2

        return int(m[len(s1) - 1][len(s2) - 1])
