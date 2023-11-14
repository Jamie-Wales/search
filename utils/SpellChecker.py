from typing import List, Tuple

import numpy as np


class SpellChecker:
    def __init__(self, input: List[str], posting_list: dict[str, List[int]]):
        self.posting_list = posting_list
        self.input = input

    def correct_words(self) -> list[str]:
        output = []
        for word in self.input:
            if word not in self.posting_list:
                closest_word, min_distance = self.find_closest_word_in_postings(word)
                if min_distance < 3:
                    print(f"Spellchecking: Changing {word} for {closest_word}")
                    output.append(closest_word)
            else:
                output.append(word)
        return output

    def find_closest_word_in_postings(self, input_word: str) -> Tuple[str, int]:
        min_distance = float('inf')
        closest_word = input_word
        for word in self.posting_list:
            distance = self.edit_distance(input_word, word)
            if distance < min_distance:
                min_distance = distance
                closest_word = word
        return closest_word, min_distance

    def edit_distance(self, s1: str, s2: str) -> int:
        # ... (same as your current edit_distance method) ...    def edit_distance(self, s1: str, s2: str) -> int:
        s1 = ' ' + s1
        s2 = ' ' + s2

        m = np.zeros((len(s1), len(s2)))

        for i in range(len(s1)):
            m[i][0] = i
        for j in range(len(s2)):
            m[0][j] = j

        for i in range(1, len(s1)):
            for j in range(1, len(s2)):
                offset = 0 if s1[i] == s2[j] else 1
                m[i][j] = min(m[i - 1][j - 1] + offset, m[i - 1][j] + 1, m[i][j - 1] + 1)

        return int(m[len(s1) - 1][len(s2) - 1])
