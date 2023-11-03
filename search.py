from search_components import CorpusManager, Document
from engine import PostingList
import pickle

file = open("CorpusManager.pkl", "rb")
manager: CorpusManager = pickle.load(file)
file.close()

pl = PostingList()

for document in manager.raw_corpus.document_list:
    for token in document.tokenised_content:
        pl.add_posting(token, document.path)

