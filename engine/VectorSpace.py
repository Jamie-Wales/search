import numpy


class VectorSpace:
    """Representation of our vector space"""
    def __init__(self, corpus_word_manager):
        self.lemmatized_vectorspace = numpy.array(list(corpus_word_manager.words['lemmatized'].keys()))
        self.stemmed_vectorspace = numpy.array(list(corpus_word_manager.words['stemmed'].keys()))
        self.original_vectorspace = numpy.array(list(corpus_word_manager.words['original'].keys()))
