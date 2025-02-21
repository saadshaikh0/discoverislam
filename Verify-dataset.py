import json
import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from rapidfuzz import fuzz, process
from openai import OpenAI  # Uncomment if using GPT for verification

# Load the datasets
quran_file = "/mnt/data/quran_instruction_test.json"
hadith_file = "/mnt/data/hadith_instruction_test.json"
tafsir_file = "/mnt/data/tafsir_instruction_test.json"

# Load Sentence Transformer Model (for similarity matching)
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Load the data
def load_data(file):
    with open(file, "r", encoding="utf-8") as f:
        return json.load(f)

quran_data = load_data(quran_file)
hadith_data = load_data(hadith_file)
tafsir_data = load_data(tafsir_file)

# Combine all QA datasets
qa_dataset = quran_data + hadith_data + tafsir_data

# Extract question-answer pairs
questions = [entry["instruction"] for entry in qa_dataset]
answers = [entry["output"] for entry in qa_dataset]

# ✅ **Step 1: Detect Duplicate Questions**
def find_duplicates(questions):
    duplicates = []
    for i, q1 in enumerate(questions):
        for j, q2 in enumerate(questions[i+1:], start=i+1):
            similarity = fuzz.ratio(q1.lower(), q2.lower())
            if similarity > 90:  # Threshold for near-duplicate detection
                duplicates.append((i, j, similarity))
    return duplicates

duplicates = find_duplicates(questions)
print(f"🔍 Found {len(duplicates)} duplicate questions!")

# ✅ **Step 2: Identify Similar Answers Using FAISS**
# Convert answers to embeddings
answer_embeddings = np.array(model.encode(answers))
index = faiss.IndexFlatL2(answer_embeddings.shape[1])
index.add(answer_embeddings)

# Find near-duplicate answers
def find_similar_answers(answers, threshold=0.85):
    similarities = []
    for i, emb in enumerate(answer_embeddings):
        D, I = index.search(np.array([emb]), k=5)  # Find top-5 nearest answers
        for j, score in zip(I[0], D[0]):
            if i != j and score < threshold:  # Lower score = more similar
                similarities.append((i, j, score))
    return similarities

similar_answers = find_similar_answers(answers)
print(f"🔍 Found {len(similar_answers)} similar answer pairs!")

# ✅ **Step 3: AI-Based Validation (Optional - Uses GPT)**
# OpenAI GPT-based correctness check
client = OpenAI(api_key="YOUR_OPENAI_API_KEY")  # Uncomment and add your API key

def check_answer_correctness(question, answer):
    prompt = f"""
    The following is a Quranic/Hadith/Tafsir-based question-answer pair. Verify if the answer is correct and relevant.

    **Question:** {question}
    **Answer:** {answer}

    **Reply with:** 'Correct' or 'Incorrect' with an explanation.
    """
    response = client.completions.create(model="gpt-4", prompt=prompt, max_tokens=100)
    return response.choices[0].text.strip()

# Uncomment to use AI validation
# validated_answers = {q: check_answer_correctness(q, a) for q, a in zip(questions, answers)}

# ✅ **Step 4: Save Identified Issues for Review**
issues = {
    "duplicates": [{"q1": questions[i], "q2": questions[j], "similarity": sim} for i, j, sim in duplicates],
    "similar_answers": [{"q1": questions[i], "a1": answers[i], "q2": questions[j], "a2": answers[j], "score": score} for i, j, score in similar_answers],
}

with open("/mnt/data/verification_results.json", "w", encoding="utf-8") as f:
    json.dump(issues, f, indent=4, ensure_ascii=False)

print("✅ Verification completed. Results saved to 'verification_results.json'")
