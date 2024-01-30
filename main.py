import os
import re
import threading
import tkinter as tk
import webbrowser
from pathlib import Path
from tkinter import font as tkFont

import ttkbootstrap as ttk
from PIL import Image, ImageTk
from ttkbootstrap.constants import *
from ttkbootstrap.scrolled import ScrolledFrame

from engine.Search import Search
from utils.SpellChecker import SpellChecker


class SearchApp:
    """
    A search application built with Tkinter.
    """

    def __init__(self, master):
        self.master = master
        master.title('Search Engine')

        self.search = Search()
        self.named_entites = self.search.ner_words
        self.spell_checker = SpellChecker(
            self.search.corpus_manager.get_raw_corpus().word_manager,
            self.search.corpus_manager.get_raw_corpus().vector_space
        )

        self.results = None
        self.user_feedback = False
        self.setup_fonts()
        self.setup_ui()
        self.checkbox_vars = {}
        self.ner_words = {}

    def setup_fonts(self):
        """Setting up fonts."""
        self.futura_bold_large = tkFont.Font(family="Futura", size=18, weight="bold")
        self.futura_medium = tkFont.Font(family="Futura", size=14)
        self.futura_small = tkFont.Font(family="Futura", size=12)

    def init_relevant_feedback(self):
        self.user_feedback = not self.user_feedback
        self.clear_results()
        if self.results is not None:
            self.display_results(self.results)
            if self.user_feedback:
                self.rerank_button.pack(side=tk.TOP, pady=5)
            else:
                self.rerank_button.pack_forget()

    def display_results(self, results):
        """Display the search results in the results frame."""
        for elements in results:
            element_frame = ttk.Frame(self.results_frame, style="light")
            element_frame.pack(side=tk.TOP, fill=tk.X, pady=5, padx=50)
            result_text = f"{elements[1].doc_id} - {elements[1].url} - {elements[0]}"
            result_label = ttk.Label(element_frame, style="light-inverse", text=result_text,
                                     font=self.futura_bold_large)
            result_label.pack(side=tk.TOP)
            body_frame = ScrolledFrame(element_frame, style="light")
            body_frame.pack(side=tk.TOP, fill=tk.X, padx=10)
            body_label = ttk.Text(body_frame, font=self.futura_medium)

            document_text = self.search.corpus_manager.get_raw_corpus().get_document_by_id(
                elements[1].doc_id).raw_content.lower()
            body_label.insert(tk.INSERT, document_text)

            # Highlight matching words in red
            body_label.tag_config("highlight", foreground="red")
            for word in elements[2]:
                word_lower = self.search.corpus_manager.get_raw_corpus().word_manager.get_word(self.word_type.get(),
                                                                                               word)
                word_lower = word_lower.__getattribute__(self.word_type.get()).lower()
                for match in re.finditer(rf"\b{re.escape(word_lower)}\b", document_text):
                    start = match.start()
                    end = match.end()
                    # Convert character offsets to Text widget indices
                    start_index = body_label.index(f"1.0+{start}c")
                    end_index = body_label.index(f"1.0+{end}c")
                    body_label.tag_add("highlight", start_index, end_index)

            body_label.config(state=tk.DISABLED)
            body_label.pack(side=tk.TOP, fill=X, padx=10)
            button_frame = ttk.Frame(element_frame)
            button_frame.pack(side=TOP, pady=5)
            view_button = ttk.Button(button_frame, text="View",
                                     command=lambda d=elements[1].url: self.view_document(d), bootstyle="dark")
            view_button.pack(side=LEFT)
            self.results = results

            if self.user_feedback:
                var = tk.BooleanVar()  # Create a control variable for the checkbox
                checkbox = tk.Checkbutton(button_frame, text='Mark Relevant', pady=10, variable=var)
                checkbox.pack(side=LEFT, padx=2)
                self.checkbox_vars[elements[1].doc_id] = var  # Map the control variable to the docID

    def clear_results(self):
        """Clear any existing results from the results.get_raw_corpus().frame."""
        for widget in self.results_frame.winfo_children():
            widget.destroy()

        for widget in self.spell_suggestion_frame.winfo_children():
            widget.destroy()
        self.results_frame.pack()
        self.checkbox_vars.clear()

    def setup_ui(self):
        """Set up the user interface."""
        # Logo setup
        logo_path = Path(__file__).parent / "./assets/logo.png"
        logo_image = Image.open(logo_path)
        logo_photo = ImageTk.PhotoImage(logo_image)
        self.logo_label = tk.Label(self.master, image=logo_photo)
        self.logo_label.image = logo_photo
        self.logo_label.pack(pady=20)

        # Top Frame for Search Input and Button
        self.top_frame = ttk.Frame(self.master)
        self.top_frame.pack(side=tk.TOP, fill=tk.X, padx=20)
        self.setup_top_frame()

        # Dropdown setup
        self.setup_dropdown()

        # List frame setup
        self.list_frame = ttk.Frame(self.master)
        self.list_frame.pack(side=tk.TOP)
        self.type_ahead = tk.Listbox(self.list_frame, width=50, height=5)
        self.type_ahead.bind('<<ListboxSelect>>', self.on_listbox_select)
        self.type_ahead.pack(side=tk.TOP)

        # Spell suggestion frame setup
        self.spell_suggestion_frame = ttk.Frame(self.master)
        self.spell_suggestion_frame.pack(side=tk.TOP, fill=tk.X)

        # Relevance feedback and results frame setup
        self.relevance_feedback = ttk.Button(
            self.master, bootstyle="light-outline",
            text="Mark relevant", command=self.init_relevant_feedback
        )
        self.relevance_feedback.pack(side=tk.TOP, pady=5)

        self.results_frame = ScrolledFrame(self.master, autohide=False)
        self.results_frame.pack(fill=BOTH, expand=YES)
        # Rerank with Feedback Button
        self.rerank_button = ttk.Button(
            self.list_frame, bootstyle="success",
            text="Rerank with Feedback", command=self.rerank_with_feedback
        )

        # Quit Button
        self.quit_button = ttk.Button(
            self.master, text="Quit", command=self.master.quit,
            style="danger-outline"
        )
        self.quit_button.pack(side=tk.BOTTOM, pady=5)

    def setup_top_frame(self):
        """Setup for the top frame of the UI."""
        self.label = ttk.Label(self.top_frame, text="Enter a query:", font=self.futura_bold_large)
        self.label.pack(side=tk.LEFT)

        self.entry = ttk.Entry(self.top_frame, font=self.futura_medium)
        self.entry.pack(padx=20, side=tk.LEFT, fill=tk.X, expand=True)
        self.entry.bind("<KeyRelease>", self.on_key_release)

        self.search_button = ttk.Button(self.top_frame, text="Search", command=self.perform_search, bootstyle="success")
        self.search_button.pack(side=tk.LEFT)

    def setup_dropdown(self):
        """Setup for the dropdown menus."""
        self.dropdown_frame = ttk.Frame(self.master)
        self.dropdown_frame.pack(side=tk.TOP, fill=tk.X, pady=20, padx=20)

        self.word_type_label = ttk.Label(self.dropdown_frame, text="Select Word Type:")
        self.word_type_label.pack(side=tk.LEFT, padx=5)

        self.word_type = ttk.Combobox(
            self.dropdown_frame, values=["original", "stemmed", "lemmatized"],
            state='readonly', bootstyle="info"
        )
        self.word_type.set("lemmatized")  # Set default value
        self.word_type.pack(side=tk.LEFT, padx=5)

        self.vec_type_label = ttk.Label(self.dropdown_frame, text="Select weighting algorithm:")
        self.vec_type_label.pack(side=tk.LEFT, padx=5)

        self.vec_type = ttk.Combobox(
            self.dropdown_frame,
            values=["TFIDFVector", "TFIDFFieldVector", "BM25plusVector", "BM25plusFieldVector"],
            state='readonly', bootstyle="info"
        )
        self.vec_type.set("BM25plusFieldVector")  # Set default value
        self.vec_type.pack(side=tk.LEFT, padx=5)

    def on_key_release(self, event):
        # Cancel previous job if it's still queued

        if hasattr(self, '_job'):
            self.master.after_cancel(self._job)

        # Schedule the update after a delay
        self._job = self.master.after(5, self.update_suggestions)

    def on_listbox_select(self, event):
        """Event handler for when an item is selected in the type_ahead Listbox."""
        # Get the index of the selected item
        selection = event.widget.curselection()
        if selection:
            index = selection[0]
            name = self.suggestions[index][0]
            category = self.suggestions[index][1]
            self.ner_words[name] = (name, category)
            if self.entry == "":
                self.entry.insert(0, f"{name} ")
            else:
                self.entry.insert(-1, f" {name} ")

    def check_ner_words(self):
        to_remove = []
        for strings in self.ner_words.keys():
            if strings not in self.entry.get():
                to_remove.append(strings)

        for remove in to_remove:
            self.ner_words.pop(remove)

    def update_suggestions(self):
        typed_text = self.entry.get()
        self.type_ahead.pack_forget()
        self.check_ner_words()
        if typed_text == "":
            return
        self.type_ahead.pack(side=tk.TOP)
        self.type_ahead.delete(0, tk.END)

        def update_listbox_with_suggestions():
            suggestions = self.search.ner_words.find_words_with_prefix(typed_text)
            # Schedule the Listbox update in the main thread
            self.master.after(0, lambda: self._update_listbox(suggestions))

        # Start a new thread for Trie lookup and updating the Listbox
        threading.Thread(target=update_listbox_with_suggestions).start()

    def update_listbox(self, suggestions):
        # Schedule the Listbox update in the main thread
        root.after(0, lambda: self._update_listbox(suggestions))

    def _update_listbox(self, suggestions):
        self.type_ahead.delete(0, tk.END)
        self.suggestions = {}
        if len(suggestions) == 0:
            self.type_ahead.pack_forget()
        for index, suggestion in enumerate(suggestions):
            self.suggestions[index] = (suggestion[0], suggestion[1])
            self.type_ahead.insert(tk.END, f"{suggestion[0]}, Type: {suggestion[1]}")
        if len(self.suggestions) != 0:
            self.type_ahead.pack()

    def get_checked_documents(self):
        """Return a list of document IDs for which the checkboxes are checked."""
        checked_docs = [doc_id for doc_id, var in self.checkbox_vars.items() if var.get()]
        return checked_docs

    def perform_search(self):
        """Perform a search operation based on the query and display results."""
        search_query = self.entry.get()
        self.clear_results()
        self.type_ahead.pack_forget()
        self.results = self.search.search(self.word_type.get(), self.vec_type.get(), search_query, self.ner_words)
        self.spell_checker.correct_words(self.search.search_input.word_manager)
        if self.spell_checker.corrected_vector is not None:
            self.display_spell_suggestion()
        if len(self.results) > 0:
            self.display_results(self.results)
        else:
            ttk.Label(self.results_frame, text="Sorry, no results found", bootstyle="danger-inverse").pack()

    def display_spell_suggestion(self):
        """Display the spell check suggestion as clickable text."""
        suggestion = "Did you mean: "
        for word in self.spell_checker.corrected_words.values():
            suggestion += f"{word.original} "

        suggestion_label = ttk.Label(self.spell_suggestion_frame, text=suggestion, font=self.futura_bold_large,
                                     style="danger-inverse",
                                     cursor="hand2")
        suggestion_label.pack(side=tk.TOP)
        suggestion_label.bind("<Button-1>", self.on_spell_correction)

    def on_spell_correction(self, event=None):
        """Handle the user's response to the spell check suggestion."""
        self.entry.delete(0, tk.END)
        self.results = self.search.rerank(self.word_type.get(), self.vec_type.get(),
                                          self.spell_checker.corrected_vector)
        self.clear_results()
        self.display_results(self.results)

    def view_document(self, url):
        """Open the document in the default web browser."""
        file_path = 'file://' + os.path.realpath(url)
        webbrowser.open(file_path)

    def rerank_with_feedback(self):
        """Rerank search results based on the user's relevance feedback."""
        checked_docs = self.get_checked_documents()
        if checked_docs:
            new_vec = self.search.relevance_feedback(self.word_type.get(), self.vec_type.get(), checked_docs)
            self.results = self.search.rerank(self.word_type.get(), self.vec_type.get(),
                                              new_vec)
            self.clear_results()
            self.display_results(self.results)


if __name__ == "__main__":
    root = ttk.Window(themename="superhero")
    root.geometry('1300x900')
    app = SearchApp(root)

    root.mainloop()
