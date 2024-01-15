import csv

import pandas as pd
from matplotlib import pyplot as plt

from engine.Search import Search

search = Search()

corp = search.corpus_manager.get_raw_corpus()
with open('count.csv', mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    for word in corp.word_manager.words["lemmatized"].values():
        if len(word.lemmatized_concurrent) != 0:

            row = [words for words in word.lemmatized_concurrent]
            row.append(word.lemmatized)
            writer.writerow(row)

        else:
            writer.writerow(word.lemmatized)
