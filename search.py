import math

import PySimpleGUI as sg
import numpy

from engine import Search, Ranker
from utils import SpellChecker


def okampi25plus(self, word, term_frequency, document_frequency, metadata, field=True, sections=None):
    if field:
        total_score = self.field(word, metadata, sections)
        tf = total_score * (1.2 + 1) / (
                total_score + 1.2 * (1 - 0.75 + 0.75 * sum(document_frequency.values()) / self.doc_length))
        return tf


def _avg_doc_length(self):
    count = 0
    sum = 0
    for doc in self.corpus.documents:
        sum += len(doc.document_frequency)
        count += 1

    return sum / count


def okampi25plusidf(self, word):
    docsHoldingN = self._get_n_docs(word)
    top = len(self.corpus.documents) - docsHoldingN
    idf = math.log(top + 0.5 / (docsHoldingN + 0.5))
    return idf


def _get_n_docs(self, word):
    count = 0
    for docs in self.corpus.documents:
        if word in docs.document_frequency:
            count += 1
    return count


def get_word_doc_matrix(self):
    # Initialize a dictionary where each word maps to a list of TF-IDF scores, one per document
    word_score = {}
    for word in self.vector_space:
        word_score[word] = []
    for doc in self.corpus.documents:
        word_score[word].append(doc.vector.raw_vec.get(word))

    return word_score


def find_most_related_word(self, target_word):
    sorted_related_words = []
    word_arr = self.word_doc_matrix.get(target_word)

    # Ensure word_arr is not None before proceeding
    if word_arr is not None:
        for keys in self.word_doc_matrix.keys():
            check = self.word_doc_matrix[keys]
            score = 0
            # Ensure check is not None before doing dot product
        if check is not None:
            check = [0 if v is None else v for v in check]
            word_arr = [0 if v is None else v for v in word_arr]
            score += numpy.dot(check, word_arr)
            sorted_related_words.append((keys, score))

    sorted_related_words = sorted(sorted_related_words, key=lambda x: x[1], reverse=True)
    top_5_words = [word for word, score in sorted_related_words if word != target_word][:5]

    return top_5_words


if __name__ == "__main__":
    search = Search()
    sg.theme('LightGreen')

    layout = [
        [sg.Text("Enter a query:", font=("Verdana", 15)), sg.Input(key='-INPUT-', expand_x=True), sg.Button('Search')],
        [sg.Column([], key="-SPELLCHECK-")],
        [sg.Column([], key='-RESULTS-COLUMN-', expand_y=True, expand_x=True)],
        [sg.Button('Quit')]
    ]

    # Create the window
    window = sg.Window('Search Engine', layout, size=(1200, 800))


    def perform_search(query):
        row = [
            sg.pin(
                sg.Col([[
                    sg.Text(f"Title: {query[1].url}"), sg.Text(f"Score: {query[0]}"),
                    sg.Button("View", key=("-View-", 1))
                ]], key="result"
                ))
        ]

        return row


    def clear_column(window, column_key):
        """
        Clear all widgets from the specified column.
        """
        for widget in window[column_key].Widget.winfo_children():
            widget.destroy()


    def check_user_spelling(spellChecker: SpellChecker):
        spell_layout = []
        for word in spellChecker.corrected_words:
            row = [
                sg.pin(
                    sg.Col([[
                        sg.Text(f"Did you mean: {word}"), sg.Button("Yes", key="-correct-")
                    ]], key="spellCheck"
                    ))
            ]

            spell_layout.append(row)
        return spell_layout


    # Event loop
    # Event loop
    while True:
        event, values = window.read()
        if event in (sg.WINDOW_CLOSED, 'Quit'):
            break
        elif event == 'Search':
            # Clear the existing results from the results column
            clear_column(window, '-RESULTS-COLUMN-')
            clear_column(window, '-SPELLCHECK-')

            # Execute the search and prepare new results
            search_query = values['-INPUT-']
            elements, spellChecker = search.search("lemmatized", search_query)
            if spellChecker.corrected_vector is not None:
                window.extend_layout(window['-SPELLCHECK-'], check_user_spelling(spellChecker))

            new_layout = []
            if len(elements) == 0:
                new_layout.append([sg.Text("No results")])
            else:
                for element in elements:
                    new_layout.append(perform_search(element))

            # Update the results column with new results
            window.extend_layout(window["-RESULTS-COLUMN-"], new_layout)
            window.refresh()

        elif event == '-correct-':
            ranker = Ranker()
            elements = ranker.tf_idf_vector("lemmatized", search.document_vector_store, spellChecker.corrected_vector)
            new_layout = []
            if len(elements) == 0:
                new_layout.append([sg.Text("No results")])
            else:
                for element in elements:
                    new_layout.append(perform_search(element))

            # Update the results column with new results
            window.extend_layout(window["-RESULTS-COLUMN-"], new_layout)
            window.refresh()

    # Close the window
    window.close()
