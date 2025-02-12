import openai
import numpy as np
import json
import faiss
import os
# Load your dataset
with open("bukhari_hadith_embeddings.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Load the FAISS index
index = faiss.read_index("bukhari_hadith_embeddings.index")
client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


# Function to generate embedding for the query
def get_embedding(text, model="text-embedding-ada-002"):
    response = client.embeddings.create(
        input=text,
        model=model
    )
    return response.data[0].embedding

# Function to search Quran and Hadith texts
def search_quran_hadith(query, top_k=3):
    query_embedding = np.array(get_embedding(query)).astype("float32").reshape(1, -1)
    distances, indices = index.search(query_embedding, top_k)

    results = []
    for i in range(top_k):
        idx = indices[0][i]
        results.append({
            "text": data[idx]["text"],
            "translation": data[idx]["translation"],
            "source": data[idx]["source"],
            "reference": data[idx]["reference"],
            "score": distances[0][i]
        })

    return results

# # Example Query
# query = "can you tell me about islam god?"
# results = search_quran_hadith(query)

# # Display Results
# for r in results:
#     print(f"Source: {r['source']} | Reference: {r['reference']}")
#     print(f"Text: {r['text']}")
#     print(f"Translation: {r['translation']}")
#     print(f"Score: {r['score']}\n")
