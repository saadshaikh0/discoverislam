import openai
import json
import numpy as np
import faiss
import os
from transformers import GPT2TokenizerFast
from tqdm import tqdm
import pickle

API_KEY = os.getenv("OPENAI_API_KEY")

# Set your OpenAI API Key
client = openai.OpenAI(api_key=API_KEY)
# Initialize tokenizer
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

# Define token limit
MAX_TOKENS = 8192  
EMBEDDING_DIM = 1536  # Embedding dimension of "text-embedding-ada-002"

def count_tokens(text):
    """Returns number of tokens in the text."""
    return len(tokenizer.encode(text))

def split_text(text, max_tokens=MAX_TOKENS):
    """Splits long text into smaller chunks within the token limit."""
    tokens = tokenizer.encode(text)
    chunks = [tokens[i:i + max_tokens] for i in range(0, len(tokens), max_tokens)]
    return [tokenizer.decode(chunk) for chunk in chunks]

def get_embeddings(text):
    """
    Returns embeddings for text. If it's long, it splits and returns multiple embeddings.
    """
    if count_tokens(text) <= MAX_TOKENS:
        response = client.embeddings.create(input=text, model="text-embedding-ada-002")
        return [response.data[0].embedding]  # Single embedding
    
    # If text is too long, split it and get multiple embeddings
    chunks = split_text(text)
    embeddings = []
    
    for chunk in chunks:
        response = client.embeddings.create(input=chunk, model="text-embedding-ada-002")
        embeddings.append(response.data[0].embedding)

    return embeddings  # List of embeddings for each chunk

# Load Tafsir dataset
with open("tafsir_data.json", "r", encoding="utf-8") as f:
    tafsir_data = json.load(f)

# FAISS Setup: Using IndexFlatL2 for fast retrieval
index = faiss.IndexFlatL2(EMBEDDING_DIM)  # L2 distance index
metadata = {}  # Dictionary to store verse references

id_counter = 0
for item in tqdm(tafsir_data, desc="Processing Tafsir"):
    verse = item.get("verse", "")
    tafsir_text = item.get("tafsir_text", "")
    reference = item.get("reference", "")

    # Get embeddings (single or multiple)
    embedding_vectors = get_embeddings(tafsir_text)

    for embedding in embedding_vectors:
        embedding_np = np.array(embedding, dtype=np.float32).reshape(1, -1)
        index.add(embedding_np)  # Add embedding to FAISS index
        
        # Store metadata with corresponding ID
        metadata[id_counter] = {
            "verse": verse,
            "tafsir": tafsir_text,
            "reference": reference
        }
        id_counter += 1

# Save FAISS index
faiss.write_index(index, "tafsir_faiss.index")

# Save metadata as a pickle file for easy retrieval
with open("tafsir_metadata.pkl", "wb") as f:
    pickle.dump(metadata, f)

print(f"Stored {id_counter} embeddings in FAISS successfully!")
