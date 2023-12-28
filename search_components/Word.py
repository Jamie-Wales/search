from nltk import PorterStemmer, WordNetLemmatizer


class Word:
    def __init__(self, word: str, tag: str, position: int, stemmer: PorterStemmer, lemmer: WordNetLemmatizer):
        self.original = word
        self.stemmed = None
        self.lemmatized = None
        self.tag = tag
        self.position = position
        self.stem_word(word, stemmer)
        self.lemmatize_word(word, lemmer)

    def stem_word(self, word: str, stemmer: PorterStemmer):
        self.stemmed = stemmer.stem(word)

    def lemmatize_word(self, word: str, lemmer: WordNetLemmatizer):
        self.lemmatized = lemmer.lemmatize(word)

    def get(self, type: str):
        return self.__getattribute__(f"{type}")


class QueryWord:
    def __init__(self, word: str, stemmer: PorterStemmer = None, lemmer: WordNetLemmatizer = None):
        self.original = word
        self.stemmed = self.stem_word(word, stemmer)
        self.lemmatized = self.lemmatize_word(word, lemmer)

    @staticmethod
    def stem_word(word: str, stemmer: PorterStemmer) -> str:
        if stemmer is None:
            stemmer = PorterStemmer()
        return stemmer.stem(word)

    @staticmethod
    def lemmatize_word(word: str, lemmar: WordNetLemmatizer) -> str:
        if lemmar is None:
            lemmar = WordNetLemmatizer()
        return lemmar.lemmatize(word)

    def get(self, type: str):
        return self.__getattribute__(f"{type}")
