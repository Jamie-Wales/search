from search_components import CorpusManager, Corpus, Document
from engine import PostingList
from utils import check_and_overwrite

dm = CorpusManager()

corpus = dm.get_raw_corpus()


## todo: fix world-tour-soccer-2005 meta data not in CSV?