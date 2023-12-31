import math
import os
import tkinter as tk
import webbrowser

import numpy
import ttkbootstrap as ttk

from engine import Search
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


"""if __name__ == "__main__":
    search = Search()
    spell_checker = SpellChecker(search.corpus_manager.word_manager, search.corpus_manager.vector_space)
    sg.theme('SystemDefault')

    layout = [
        [sg.Text("Enter a query:", font=("Futura", 18, "bold"), pad=(20, 20)),
         sg.Input(key='-INPUT-', expand_x=True, font=("Futura", 16), pad=(20, 20)),
         sg.Button('Search', font=("Futura", 16), pad=(20, 20), button_color=("white", "green"))],
        [sg.Column([], key="-SPELLCHECK-", pad=(20, 20))],
        [sg.Column([], key='-RESULTS-COLUMN-', expand_y=True, expand_x=True, pad=(20, 20))],
        [sg.Canvas(key='-CANVAS-')],
        [sg.Button('Quit', font=("Futura", 16), pad=(20, 20), button_color=("white", "red"))]
    ]

    window = sg.Window('Search Engine', layout, size=(1300, 900))
    canvas_elem = window['-CANVAS-']
    canvas = canvas_elem.Widget
    frame = tk.Frame(canvas)
    canvas.create_window((0, 0), window=frame, anchor='nw')

    def perform_search(query):
        row = [
            sg.pin(
                sg.Col([[
                    sg.Text(f"Title: {query[1].url}", font=("Futura", 16)),
                    sg.Text(f"Score: {query[0]}", font=("Futura", 16)),
                    sg.Button("View", key=(f"-View-{query[1].doc_id}"), font=("Futura", 16),
                              button_color=("white", "blue"))
                ]], key=f"result-{query[1].doc_id}", pad=(10, 10)
                ))
        ]
        return row


    def clear_column(window, column_key):
        for widget in window[column_key].Widget.winfo_children():
            widget.destroy()


    def check_user_spelling(spell_checker: SpellChecker):
        # Enhanced styling for the spell check suggestion row
        spell_layout = []
        output = "Did you mean: "
        for word in spell_checker.corrected_words.values():
            output += f"{word.original} "

        unique_key = f"-correct-{int(time())}"
        row = [
            sg.pin(
                sg.Col([[
                    sg.Text(output, font=("Futura", 16)),
                    sg.Button("Yes", key=unique_key, font=("Futura", 16), button_color=("white", "orange"))
                ]], pad=(10, 10)
                ))
        ]
        spell_layout.append(row)
        return spell_layout


    # Event loop
    while True:
        event, values = window.read()
        if event in (sg.WINDOW_CLOSED, 'Quit'):
            break
        elif event == 'Search':
            clear_column(window, '-RESULTS-COLUMN-')
            clear_column(window, '-SPELLCHECK-')
            window['-CANVAS-'].update(visible=False)

            search_query = values['-INPUT-']
            elements = search.search("lemmatized", search_query)
            spell_checker.correct_words(search.search_input.word_manager)
            if spell_checker.corrected_vector is not None:
                window.extend_layout(window['-SPELLCHECK-'], check_user_spelling(spell_checker))

            new_layout = []
            if len(elements) == 0:
                new_layout.append([sg.Text("No results", font=("Futura", 16))])
            else:
                for element in elements:
                    new_layout.append(perform_search(element))
            window.extend_layout(window["-RESULTS-COLUMN-"], new_layout)
            window.refresh()

        elif event.startswith('-correct-'):
            clear_column(window, '-RESULTS-COLUMN-')
            clear_column(window, '-SPELLCHECK-')
            elements = search.rerank("lemmatized", spell_checker.corrected_vector)
            new_layout = []
            if len(elements) == 0:
                new_layout.append([sg.Text("No results", font=("Futura", 16))])
            else:
                for element in elements:
                    new_layout.append(perform_search(element))

            window.extend_layout(window["-RESULTS-COLUMN-"], new_layout)
            window.refresh()


        elif event.startswith('-View-'):
            doc_id = int(event.split('-')[-1])
            document = search.corpus_manager.get_document_by_id(doc_id)
            print(f"Document ID: {doc_id}, Document: {document}")
            window['-CANVAS-'].update(visible=True)

    window.close()

"""


class SearchApp:
    """
    A search application built with Tkinter.
    """

    def __init__(self, master):
        """Initialize the SearchApp with a master Tkinter window."""
        self.master = master
        master.title('Search Engine')

        self.search = Search()
        self.spell_checker = SpellChecker(self.search.corpus_manager.word_manager,
                                          self.search.corpus_manager.vector_space)

        self.setup_ui()

    def display_results(self, results):
        """Display the search results in the results frame."""
        for elements in results:
            # Assuming elements[1] has attributes like doc_id and url
            result_text = f"{elements[1].doc_id} - {elements[1].url}"
            result_label = ttk.Label(self.results_frame, text=result_text, font=("Futura", 16))
            result_label.pack()

            view_button = ttk.Button(self.results_frame, text="View",
                                     command=lambda d=elements[1].url: self.view_document(d))
            view_button.pack()
            result_checkbox = ttk.Checkbutton(self.results_frame)

    def clear_results(self):
        """Clear any existing results from the results frame."""
        for widget in self.results_frame.winfo_children():
            widget.destroy()

        for widget in self.spell_suggestion_frame.winfo_children():
            widget.destroy()

        self.results_frame.pack()
        self.results_frame.pack()
    def setup_ui(self):
        """Set up the user interface for the search application."""
        # Top Frame for Search Input and Button
        self.top_frame = tk.Frame(self.master)
        self.top_frame.pack(pady=20)

        self.label = ttk.Label(self.top_frame, style="light", text="Enter a query:", font=("Futura", 18, "bold"))
        self.label.pack(side=tk.LEFT, padx=10)

        self.entry = tk.Entry(self.top_frame, font=("Futura", 16), width=50)
        self.entry.pack(side=tk.LEFT, padx=10)

        self.search_button = ttk.Button(self.top_frame, style="success", text="Search", command=self.perform_search)
        self.search_button.pack(side=tk.LEFT, padx=10)

        self.spell_suggestion_frame = tk.Frame(self.master)
        self.spell_suggestion_frame.pack(side=tk.TOP, pady=10)
        self.results_frame = tk.Frame(self.master)
        self.results_frame.pack(pady=10)

        self.relevance_feedback = tk.Button(self.master, text="init_mark_relevant")

        self.quit_button = tk.Button(self.master, text="Quit", font=("Futura", 16), command=self.master.quit)
        self.quit_button.pack(pady=10)


    def perform_search(self):
        """Perform a search operation based on the query and display results."""
        search_query = self.entry.get()
        self.relevance_feedback.pack()
        self.clear_results()
        results = self.search.search("lemmatized", search_query)
        self.spell_checker.correct_words(self.search.search_input.word_manager)
        if self.spell_checker.corrected_vector is not None:
            self.display_spell_suggestion()
        self.display_results(results)


    def display_spell_suggestion(self):
        """Display the spell check suggestion as clickable text."""
        suggestion = "Did you mean: "
        for word in self.spell_checker.corrected_words.values():
            suggestion += f"{word.original} "

        suggestion_label = tk.Label(self.spell_suggestion_frame, text=suggestion, font=("Futura", 16), fg="blue",
                                    cursor="hand1")
        suggestion_label.pack(side=tk.LEFT)
        suggestion_label.bind("<Button-1>", self.on_spell_correction)

    def on_spell_correction(self, event=None):
        """Handle the user's response to the spell check suggestion."""
        self.entry.delete(0, tk.END)
        results = self.search.rerank("lemmatized", self.spell_checker.corrected_vector)
        self.clear_results()
        self.display_results(results)

    def view_document(self, url):
        """Open the document in the default web browser."""
        file_path = 'file://' + os.path.realpath(url)
        webbrowser.open(file_path)


if __name__ == "__main__":
    root = ttk.Window(themename="superhero")
    root.geometry('1300x900')
    app = SearchApp(root)
    root.mainloop()
