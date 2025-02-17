import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    BitsAndBytesConfig
)
import os
import sys

# Import search functions
sys.path.append(os.path.dirname(__file__))  
from semantic_search import search_quran_hadith, search_quran, search_tafsir

# Load Model Name
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

# Adjust BitsAndBytes Config for Lower Memory
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16  # Ensure this matches
)

# Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)


model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    quantization_config=bnb_config,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
    max_memory={0: "6GB", "cpu": "30GB"}
)


# Create Pipeline with Optimized Parameters
chat_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
   
)

MAX_TAFSIR_TOKENS = 3048  # Limit Tafsir tokens to prevent overflow

def truncate_text(text, max_tokens=MAX_TAFSIR_TOKENS):
    tokens = tokenizer.encode(text, truncation=True, max_length=max_tokens)
    return tokenizer.decode(tokens)

def build_context_from_embeddings(query: str, top_k: int = 1) -> str:
    """
    Searches FAISS indices for Quranic verses, Hadith texts, and Tafsir explanations
    to build a knowledge-based response.
    """
    quran_results = search_quran(query, top_k=top_k)
    hadith_results = search_quran_hadith(query, top_k=top_k)
    tafsir_results = search_tafsir(query, top_k=top_k)

    context_str = ""

    # Add Quranic References
    if quran_results:
        context_str += "**📖 Quranic References:**\n"
        for r in quran_results:
            context_str += f"📜 **Reference:** {r['reference']}\n🔹 {r['text']}\n\n"

    # Add Hadith References
    if hadith_results:
        context_str += "**📜 Hadith References:**\n"
        for r in hadith_results:
            context_str += f"📖 **Source:** {r['source']} | 📜 {r['reference']}\n🔹 {r['text']}\n\n"

    # Add Tafsir Explanations (Truncated)
    if tafsir_results:
        context_str += "**📘 Tafsir Explanations (Truncated):**\n"
        for r in tafsir_results:
            truncated_tafsir = truncate_text(r['tafsir'])  # Truncate Tafsir
            context_str += f"📖 **Verse:** {r['verse']} | 📜 {r['reference']}\n🔹 {truncated_tafsir}\n\n"

    return context_str.strip() if context_str else "No relevant context found."

### 🔹 Optimized Response Generation ###
def generate_response(user_query: str):
    """
    Generates an AI response incorporating knowledge from Quran, Hadith, and Tafsir.
    """
    # Build Context
    retrieved_context = build_context_from_embeddings(user_query, top_k=1)

    system_prompt = (
        "You are a knowledgeable Islamic scholar. Provide concise and factual explanations, "
        "citing references where possible.\n\n"
        "📖 **Context from Quran, Hadith, and Tafsir:**\n"
        f"{retrieved_context}\n\n"
    )
    
    prompt = f"{system_prompt}👤 User: {user_query}\n🤖 Scholar:"

    # Free Up CUDA Memory Before Generation
    torch.cuda.empty_cache()

    # Generate Response (Optimized for OOM Issues)
    response = chat_pipeline(
        prompt,
        max_new_tokens=1024,  # Lowered from 2000 to prevent OOM
        temperature=0.4,  # More deterministic
        top_p=0.8,  # Reduced sampling range
        do_sample=True
    )

    return response[0]["generated_text"]

if __name__ == "__main__":
    # Example Query
    user_query = 'What is the ruling on cryptocurrency in Islam?'
    result = generate_response(user_query)
    print(result)
