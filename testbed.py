import csv

from engine.Search import Search

import pandas as pd
import matplotlib.pyplot as plt
search_obj = Search()
# And a list of queries
vec_type = ["DocID", "TFIDFVectorSum", "TFIDFFieldVectorSum", "BM25plusVectorSum", "BM25plusFieldVectorSum"]

if True:
    with open('doc_vec.csv', mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(vec_type)
        for vector_store in search_obj.document_vector_store.document_vectors:
            tfidf_vector_avg = sum(vector_store.TFIDFVector.lemmatized_data.value.values()) / len(
                vector_store.TFIDFVector.lemmatized_data.value)
            tfidffield_vector_avg = sum(vector_store.TFIDFFieldVector.lemmatized_data.value.values()) / len(
                vector_store.TFIDFFieldVector.lemmatized_data.value)
            bm25plus_vector_avg = sum(vector_store.BM25plusVector.lemmatized_data.value.values()) / len(
                vector_store.BM25plusVector.lemmatized_data.value)
            bm25plusfield_vector_avg = sum(vector_store.BM25plusFieldVector.lemmatized_data.value.values()) / len(
                vector_store.BM25plusFieldVector.lemmatized_data.value)

            writer.writerow([
                vector_store.TFIDFVector.metadata.doc_id,
                tfidf_vector_avg,
                tfidffield_vector_avg,
                bm25plus_vector_avg,
                bm25plusfield_vector_avg
            ])
df = pd.read_csv("doc_vec.csv")

# Creating scatter plots
plt.figure(figsize=(12, 9))

# Plotting all vector types on the same chart
plt.scatter(df['DocID'], df['TFIDFVectorSum'], color='#274666', label='TFIDFVectorSum')
plt.scatter(df['DocID'], df['TFIDFFieldVectorSum'], color='#143C3C', label='TFIDFFieldVectorSum')
plt.scatter(df['DocID'], df['BM25plusVectorSum'], color='#FF3131', label='BM25plusVectorSum')
plt.scatter(df['DocID'], df['BM25plusFieldVectorSum'], color='purple', label='BM25plusFieldVectorSum')

# Calculating and plotting average lines
avg_TFIDFVectorSum = df['TFIDFVectorSum'].mean()
avg_TFIDFFieldVectorSum = df['TFIDFFieldVectorSum'].mean()
avg_BM25plusVectorSum = df['BM25plusVectorSum'].mean()
avg_BM25plusFieldVectorSum = df['BM25plusFieldVectorSum'].mean()

plt.axhline(y=avg_TFIDFVectorSum, color='#274666', linestyle='--', label='Avg TFIDFVectorSum')
plt.axhline(y=avg_TFIDFFieldVectorSum, color='#143C3C', linestyle='--', label='Avg TFIDFFieldVectorSum')
plt.axhline(y=avg_BM25plusVectorSum, color='#FF3131', linestyle='--', label='Avg BM25plusVectorSum')
plt.axhline(y=avg_BM25plusFieldVectorSum, color='purple', linestyle='--', label='Avg BM25plusFieldVectorSum')

plt.xlabel('DocID')
plt.ylabel('Avg Word Score')
plt.legend()

plt.show()
queries = [
    "Action-Adventure Games",
    "RPG Games for PlayStation",
    "The Guy Game",
    "Spyder-Man1",
    "star-wars-battlefront",
    "Iron Man",
    "James Earl Cash",
    "Crazy Taxi",
    "James Bond",
    "The Lord of the Rings: The Two Towers (PS2)",
    "Crime Game",
    "Swords and fighting",
    "Fun adventure with fantasy characters",
    "Super hero game marvel or DC",
    "Final fantasy"
]

# Word types to test

df =pd.read_csv("search_results.csv")



grouped = df.groupby(['Query', 'vec type'])

# Function to get unique URLs for each group
def get_unique_urls(group):
    return set(group['url'])


unique_urls = {}
for name, group in grouped:
    unique_urls[name] = set(group['url'])

# Apply the function to each group
# Initialize a dictionary to store URLs unique to each vectype
unique_urls_per_vectype = {}

for (query, vectype), urls in unique_urls.items():
    other_vectypes = {vt for (q, vt) in unique_urls if q == query and vt != vectype}

    # Initialize unique URLs for this vectype
    unique_urls_per_vectype[(query, vectype)] = urls.copy()

    # Remove URLs that appear in other vectypes
    for other_vectype in other_vectypes:
        unique_urls_per_vectype[(query, vectype)] -= unique_urls[(query, other_vectype)]


for (query, vectype), urls in unique_urls_per_vectype.items():
    if urls:
        print(f"Query: {query}, Vectype: {vectype}, Unique URLs: {urls}")
