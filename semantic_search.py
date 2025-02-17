import openai
import numpy as np
import json
import faiss
import os
import pickle

# Load your dataset
with open("bukhari_hadith_embeddings.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Load the FAISS index
index = faiss.read_index("bukhari_hadith_embeddings.index")
API_KEY = os.getenv("OPENAI_API_KEY")

# Set your OpenAI API Key
client = openai.OpenAI(api_key=API_KEY)
FAISS_INDEX_FILE = "quran_embeddings.index"
PASSAGES_FILE = "quran_passages.json"

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


def search_quran(query, top_k=3):
    """Search Quranic passages using FAISS"""
    # Load FAISS index
    if not os.path.exists(FAISS_INDEX_FILE):
        print("❌ FAISS index not found. Run the embedding process first!")
        return []

    index = faiss.read_index(FAISS_INDEX_FILE)

    # Load passage metadata
    with open(PASSAGES_FILE, "r", encoding="utf-8") as f:
        passages = json.load(f)

    # Get query embedding
    query_embedding = np.array(get_embedding(query)).astype("float32").reshape(1, -1)

    # Search in FAISS
    distances, indices = index.search(query_embedding, top_k)

    results = []
    for i in range(top_k):
        idx = indices[0][i]
        results.append({
            "text": passages[idx]["text"],
            "reference": passages[idx]["reference"],
            "score": distances[0][i]
        })

    return results


def search_tafsir(query, top_k=3):
    """Searches for the most relevant Tafsir passages."""
    index = faiss.read_index("tafsir_faiss.index")
    with open("tafsir_metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
    query_embedding = np.array(get_embedding(query)).astype("float32").reshape(1, -1)
    distances, indices = index.search(query_embedding, top_k)  # FAISS search

    results = []
    for i in range(top_k):
        idx = indices[0][i]
        if idx in metadata:
            results.append({
                "score": float(distances[0][i]),
                "verse": metadata[idx]["verse"],
                "tafsir": metadata[idx]["tafsir"],
                "reference": metadata[idx]["reference"]
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
