from engine import PostingList
from search_components import CorpusManager
from utils import DocumentProcessor

dm = CorpusManager()
dm.sort_corpus()
rc = dm.get_raw_corpus()
dp = DocumentProcessor()
pl = PostingList()
stemmed_pl = None
pl_stem = PostingList(False)
pl_lem = PostingList(False)

