from search_components import CorpusManager, Document
from engine import PostingList

manager = CorpusManager()

corp = manager.get_raw_corpus()

text = corp.document_list
pl = PostingList()

for document in text:
    elements = Document.tokenise(document.text_content)


for elem in elements:


print(pl)
