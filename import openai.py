import openai 
from openai import OpenAI
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

client = OpenAI()

def get_embedding(text, model="text-embedding-3-small"):
    response = client.embeddings.create(
        input=[text],
        model=model
    )
    return response.data[0].embedding

texts = ["I am a Computer Science student at the University of Washington", 
         "I am a programmer", 
         "I mainly use Python and Java." ]
records = []

for text in texts:
    embedding = get_embedding(text)
    records.append({"text" : text, "embedding" : embedding})

    df = pd.DataFrame(records)
    table = pa.Table.from_pandas(df)
    pq.write_table(table, "embeddings.parquet")

df = pd.read_parquet("embeddings.parquet")

def get_top_k_matches(query, k=3):
    query_embedding = np.array(get_embedding(query)).reshape(1, -1)
    stored_embedding = np.vstack(df['embedding'].values)
    similarities = cosine_similarity(query_embedding, stored_embedding)[0]
    top_k_indices = similarities.argsort()[-k:][::-1]
    top_matches = df.iloc[top_k_indices].copy()
    top_matches['similarity'] = similarities[top_k_indices]
    return top_matches

query = "What school do I go to?"
matches = get_top_k_matches(query, k=3)

for i, row in matches.iterrows():
    print(f"Matched: {row['text']} (score: {row['similarity']:.4f})") 
