from search_components import CorpusManager

manager = CorpusManager()

corp = manager.get_raw_corpus()

for docs in corp.document_list:
    print(docs.text_content)