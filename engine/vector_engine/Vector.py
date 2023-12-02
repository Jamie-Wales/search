import numpy

from engine import PostingList
from search_components import Corpus


class Vector:
    def __init__(self):
        self.raw_vec = {}
        self.intersection = None

