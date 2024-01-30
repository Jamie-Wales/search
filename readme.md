# Search Engine
*built by Jamie Wales*

``Tested for python 3.12``


## Overview
This search engine is built using Python and Tkinter providing a GUI for users to perform search.

**Features**
- NER
- Spell Checking
- Relevance feeedback
- Query expansion
- Swtich between vector types and word prerpocessing



## Running

To run the search engine please run the main python file

```
python main.py
```
Please note the following configuration details. Ensure the folder zipped pklfiles folder is unziped and in the root directory.
The code is looking for `./pklfiles/specific_pickle` when loading. If for whatever reason the files have been deleted. See the following

If a `Corpus` cannot be found at `./pklfiles/CorpusManager` then the code will regenerate the Corpus and corpus manager. Please ensure the `./dataset` folder is in the root directory.

If the engine cannot find `./pklfiles/document-vectors` it will regenrate them.

***
***IMPORTANT NOTE ON REGENERATION***

If you regenerated the Words will not contain the co-occuring words. If this is required. Please  delete `NER.pkl` as this will inflate the co-occuring words. Retokenise by deleting `Corpus Manager` generate a corpus and a vector space then generate a co-occuring word matrix with `DocumentVectorStore()` method `gen_word_matrix`.


Pass that matrix to the `Corpus()` `WordManager()` with the method `generate_word_matrix(self, matrix)` which passes the matrix to the word manager then call. `generate_concurrent_words` this will generate the co-occuring words. Then add the raw corpus to `self.corpus_manager.raw_corpus`, and save the Corpus Manager to "./pklfiles/CorpusManager.pkl"
```python
class Search:
    """Our main search engine class"""

    def __init__(self):
        self.corpus_manager = CorpusManager()
        self.document_vector_store = DocumentVectorStore()
        raw_corp = self.corpus_manager.get_raw_corpus()
        if self.document_vector_store.need_vector_generation:
            self.document_vector_store.generate_vectors(self.corpus_manager.get_raw_corpus())
            
        raw_corp.word_manager.generate_word_matrix(self.document_vector_store.gen_word_matrix(raw_corp))
        raw_corp.word_manager.generate_concurrent_words()
        self.corpus_manager.raw_corpus = raw_corp
        check_and_overwrite("./pklfiles/CorpusManager.pkl", self.corpus_manager)
        
        self.spellVec = None
        self.lemmar = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        self.search_input = None
```

Then you can re run main.py, but delete the ./pklfiles/document-vectors.pkl as you'll want to regen these with NER and save the new Corpus with the added terms. You can then on yournext run do the default config which is.
```Python
class Search:
    """Our main search engine class"""

    def __init__(self):
        self.corpus_manager = CorpusManager()
        self.document_vector_store = DocumentVectorStore()
        raw_corp = self.corpus_manager.get_raw_corpus()
        self.named_entity_recogniser = NamedEntityRecogniser(raw_corp)
        self.corpus_manager.raw_corpus = raw_corp
        check_and_overwrite("./pklfiles/CorpusManager.pkl")
        
        if self.document_vector_store.need_vector_generation:
            self.document_vector_store.generate_vectors(self.corpus_manager.get_raw_corpus())


        self.spellVec = None
        self.lemmar = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        self.search_input = None

```