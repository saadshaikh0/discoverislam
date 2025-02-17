import json
import openai
import faiss
import numpy as np
import os


API_KEY = os.getenv("OPENAI_API_KEY")

# Set your OpenAI API Key
client = openai.OpenAI(api_key=API_KEY)

# Configuration
CHUNK_SIZE = 7  # Number of Ayahs per chunk
QURAN_FILE = "quran_data.json"
FAISS_INDEX_FILE = "quran_embeddings.index"
PASSAGES_FILE = "quran_passages.json"

### STEP 1: Load Quran Data ###
def load_quran_data(filename):
    """Load Quranic Ayahs from JSON file"""
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)

### STEP 2: Chunk Ayahs into Passages ###
def chunk_ayahs(quran_data, chunk_size=CHUNK_SIZE):
    """Groups Quranic Ayahs into passages of `chunk_size` verses"""
    passages = []
    for i in range(0, len(quran_data), chunk_size):
        chunk = quran_data[i:i + chunk_size]
        arabic_text = " ".join([ayah["arabic_text"] for ayah in chunk])
        translation = " ".join([ayah["translation"] for ayah in chunk])
        reference = ", ".join([ayah["reference"] for ayah in chunk])

        passages.append({
            "text": f"Arabic: {arabic_text}\nTranslation: {translation}",
            "reference": reference
        })
    
    return passages

### STEP 3: Generate Embeddings ###
def get_embedding(text):
    """Get embedding from OpenAI's text-embedding-ada-002"""
    response = client.embeddings.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response.data[0].embedding

### STEP 4: Store Embeddings in FAISS ###
def store_embeddings_in_faiss(passages, faiss_index_file, passages_file):
    """Embeds the passages and stores them in FAISS"""
    embeddings = []
    
    for i, passage in enumerate(passages):
        print(f"Embedding passage {i+1}/{len(passages)}...")
        passage["embedding"] = get_embedding(passage["text"])
        embeddings.append(passage["embedding"])

    # Convert embeddings to NumPy array
    embeddings = np.array(embeddings).astype("float32")

    # Create FAISS index
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    # Save FAISS index
    faiss.write_index(index, faiss_index_file)

    # Save passages metadata
    with open(passages_file, "w", encoding="utf-8") as f:
        json.dump(passages, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Stored {len(passages)} embeddings in FAISS and saved passages!")

### MAIN FUNCTION ###
def main():
    print("🔹 Loading Quran Data...")
    quran_data = load_quran_data(QURAN_FILE)

    print("🔹 Chunking Ayahs into passages...")
    quran_passages = chunk_ayahs(quran_data, CHUNK_SIZE)

    print("🔹 Creating embeddings and storing them in FAISS...")
    store_embeddings_in_faiss(quran_passages, FAISS_INDEX_FILE, PASSAGES_FILE)

    print("\n✅ Quran Embedding Pipeline Completed!")

if __name__ == "__main__":
    main()
