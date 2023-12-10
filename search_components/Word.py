from nltk.stem import PorterStemmer, WordNetLemmatizer


class Word:
    def __init__(self, word: str, tag: str, position: int, stemmer: PorterStemmer, lemmer: WordNetLemmatizer):
        self.original = word
        self.stemmed = self.stem_word(word, stemmer)
        self.lemmatized = self.lemmatize_word(word, lemmer)
        self.tag = tag
        self.position = position

    @staticmethod
    def stem_word(word: str, stemmer: PorterStemmer) -> str:
        return stemmer.stem(word)

    @staticmethod
    def lemmatize_word(word: str, lemmar: WordNetLemmatizer) -> str:
        return lemmar.lemmatize(word)

    def get(self, type: str):
        if type is "stemmed":
            return self.stemmed
        if type is "lemmatized":
            return self.lemmatized
        if type is "original":
            return self.original


class QueryWord:
    def __init__(self, word: str, stemmer: PorterStemmer, lemmer: WordNetLemmatizer):
        self.original = word
        self.stemmed = self.stem_word(word, stemmer)
        self.lemmatized = self.lemmatize_word(word, lemmer)

    @staticmethod
    def stem_word(word: str, stemmer: PorterStemmer) -> str:
        return stemmer.stem(word)

    @staticmethod
    def lemmatize_word(word: str, lemmar: WordNetLemmatizer) -> str:
        return lemmar.lemmatize(word)

    def get(self, type: str):
        return self.__getattribute__(f"{type}")
