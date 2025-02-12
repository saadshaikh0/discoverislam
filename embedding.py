import openai
import json
import numpy as np
from transformers import GPT2TokenizerFast
import os

# Initialize OpenAI client with API key (new interface)
client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Initialize tokenizer to count tokens (compatible with OpenAI's tokenizer)
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")


# Function to count tokens in the text
def count_tokens(text):
    return len(tokenizer.encode(text))


# Function to get embeddings (using the updated OpenAI API)
def get_embedding(text, model="text-embedding-ada-002", max_tokens=8000):
    # Check if text exceeds token limit and truncate if necessary
    if count_tokens(text) > max_tokens:
        tokens = tokenizer.encode(text)
        truncated_text = tokenizer.decode(tokens[:max_tokens])
        text = truncated_text

    # Call the embedding API using the client object
    response = client.embeddings.create(
        input=text,
        model=model
    )
    return response.data[0].embedding


# Load Hadith dataset
with open("sahih_bukhari_hadith.json", "r", encoding="utf-8") as f:
    hadith_data = json.load(f)

processed_data = []
for item in hadith_data:
    english_text = item.get("English Text", "")
    arabic_text = item.get("Arabic Text", "")
    narrator = item.get("Narrator", "")
    reference = item.get("Reference", "")

    # Combine text fields to create embedding input
    combined_text = f"{narrator}\n{english_text}\n{arabic_text}"
    embedding_vector = get_embedding(combined_text)

    # Append processed data
    processed_data.append({
        "text": english_text,
        "translation": arabic_text,
        "narrator": narrator,
        "reference": reference,
        "embedding": embedding_vector,
        "source": "Sahih Bukhari"
    })

# Save processed data with embeddings
with open("bukhari_hadith_embeddings.json", "w", encoding="utf-8") as f:
    json.dump(processed_data, f, ensure_ascii=False, indent=4)

print("Bukhari hadith embeddings generated and saved successfully!")
