import numpy

from search_components.WordManager import CorpusWordManager


class VectorSpace:
    def __init__(self, corpus_word_manager: CorpusWordManager):
        self.lemmatized_vectorspace = numpy.array(list(corpus_word_manager.words['lemmatized'].keys()))
        self.stemmed_vectorspace = numpy.array(list(corpus_word_manager.words['stemmed'].keys()))
        self.original_vectorspace = numpy.array(list(corpus_word_manager.words['original'].keys()))
