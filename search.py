from engine import SearchInterface
from search_components import CorpusManager
from utils import check_and_overwrite

dm = CorpusManager()

engine = SearchInterface()

for elements in dm.get_raw_corpus().documents:
    print(elements.vector.raw_vec)


check_and_overwrite("./CorpusManager.pkl", dm)