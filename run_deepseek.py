from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    BitsAndBytesConfig
)
import os
import sys

# Import the search function from semantic_search.py
sys.path.append(os.path.dirname(__file__))  # Ensure the module can be found
from semantic_search import search_quran_hadith

model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

# BitsAndBytes config for 4-bit quant
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    quantization_config=bnb_config,
    trust_remote_code=True
)

# Create pipeline
chat_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer
)

def build_context_from_embeddings(query: str, top_k: int = 3) -> str:
    """
    Search the FAISS index using the user query and build
    a string containing the top K retrieved hadith texts/translations.
    """
    results = search_quran_hadith(query, top_k=top_k)
    context_str = ""
    for r in results:
        # Optionally format the retrieved data
        context_str += f"Source: {r['source']} | Reference: {r['reference']}\n"
        context_str += f"Text (Arabic/Original): {r['text']}\n"
        context_str += f"Translation: {r['translation']}\n\n"
    return context_str.strip()

def generate_response(user_query: str):
    """
    Generate an answer using the DeepSeek model, incorporating the
    relevant context from the FAISS index search into the system prompt.
    """
    # Build context from embeddings
    retrieved_context = build_context_from_embeddings(user_query, top_k=2)

    system_prompt = (
        "You are a knowledgeable Islamic scholar. Provide concise, factual, and "
        "clear explanations without unnecessary extra thought. If you use the context, "
        "cite it in your answer.\n\n"
        "Context from Hadith:\n"
        f"{retrieved_context}\n\n"
    )
    prompt = f"{system_prompt}User: {user_query}\nAnswer:"

    import torch
    torch.cuda.empty_cache()
    # Generate response
    response = chat_pipeline(
        prompt,
        max_new_tokens=2000,
        temperature=0.5,
        do_sample=True,
        top_p=0.9
    )
    return response[0]["generated_text"]

if __name__ == "__main__":
    # Example user query
    user_query = "will jesus be coming back?"
    result = generate_response(user_query)
    print(result)