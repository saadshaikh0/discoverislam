import faiss
import numpy as np
import json
# Load embeddings
with open("bukhari_hadith_embeddings.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Convert embeddings to NumPy array
embeddings = np.array([item["embedding"] for item in data]).astype("float32")

# Create FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])  # L2 distance
index.add(embeddings)

# Save FAISS index
faiss.write_index(index, "bukhari_hadith_embeddings.index")
print("FAISS index saved!")
