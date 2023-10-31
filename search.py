from search_engine import CorpusManager
manager = CorpusManager()


for item in manager.raw_corpus.document_list:
    print(item.text_content)
