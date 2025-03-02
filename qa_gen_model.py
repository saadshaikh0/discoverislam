from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
import json
import re
import os

# Load Mistral-7B-Instruct-v0.3 model and tokenizer
model_name = "mistralai/Mistral-7B-Instruct-v0.3"
quantization_config = BitsAndBytesConfig(load_in_4bit=True)

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quantization_config)

qa_pipeline = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

# Function to remove HTML tags
def remove_html_tags(text):
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)

# Function to generate question-answer pairs
def generate_qa(text):
    prompt = f"Generate three question-answer pairs based on the following Islamic text:\n\n{text}\n\nQuestions and Answers:"
    response = qa_pipeline(prompt, max_new_tokens=1024, do_sample=True, temperature=0.7)
    return response[0]["generated_text"]

# Function to format QA output
def process_qa_output(qa_pairs):
    qa_list = qa_pairs.split("\n")
    formatted_qa = []
    current_question = None
    current_answer = ""

    for line in qa_list:
        line = line.strip()
        if not line or line.lower() in ["questions and answers:", ""]:
            continue
        if line.lower().startswith("question:") or re.match(r"^\d+\. ", line):
            if current_question and current_answer:
                formatted_qa.append({"instruction": current_question, "output": current_answer.strip()})
            current_question = re.sub(r"^\d+\.\s?", "", line.replace("Question:", "").strip())
            current_answer = ""
        elif line.lower().startswith("answer:"):
            current_answer = line.replace("Answer:", "").strip()
        elif current_question and current_answer:
            current_answer += " " + line

    if current_question and current_answer:
        formatted_qa.append({"instruction": current_question, "output": current_answer.strip()})

    return formatted_qa

# Function to save progress
def save_progress(data, filename):
    with open(filename, "w", encoding="utf-8") as outfile:
        json.dump(data, outfile, indent=4, ensure_ascii=False)

# -------------------- 1. Generate QA from Tafsir --------------------
def generate_qa_from_tafsir():
    input_file = "tafsir_data.json"
    output_file = "tafsir_instruction_test.json"

    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as outfile:
            try:
                formatted_qa = json.load(outfile)
            except json.JSONDecodeError:
                formatted_qa = []
    else:
        formatted_qa = []

    with open(input_file, "r", encoding="utf-8") as file:
        tafsir_data = json.load(file)

    batch_size = 100
    for idx, tafsir_entry in enumerate(tafsir_data):
        cleaned_text = remove_html_tags(tafsir_entry["tafsir_text"])
        max_length = 5000
        encoded_text = tokenizer.encode(cleaned_text, truncation=True, max_length=max_length, return_tensors="pt")
        truncated_text = tokenizer.decode(encoded_text[0], skip_special_tokens=True)

        qa_pairs = generate_qa(truncated_text)
        formatted_qa.extend(process_qa_output(qa_pairs))

        if (idx + 1) % batch_size == 0:
            save_progress(formatted_qa, output_file)
            print(f"Saved progress at Tafsir entry {idx + 1}")

    save_progress(formatted_qa, output_file)
    print("Tafsir QA generation completed.")

# -------------------- 2. Generate QA from Hadith --------------------
def generate_qa_from_tafsir_maarif():
    input_file = "tafsir_data_maarif-ul-quran.json"
    output_file = "tafsir_maarif_instruction_test.json"

    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as outfile:
            try:
                formatted_qa = json.load(outfile)
            except json.JSONDecodeError:
                formatted_qa = []
    else:
        formatted_qa = []

    with open(input_file, "r", encoding="utf-8") as file:
        tafsir_data = json.load(file)

    batch_size = 10
    for idx, tafsir_entry in enumerate(tafsir_data):
        cleaned_text = remove_html_tags(tafsir_entry["tafsir_text"])
        max_length = 5000
        encoded_text = tokenizer.encode(cleaned_text, truncation=True, max_length=max_length, return_tensors="pt")
        truncated_text = tokenizer.decode(encoded_text[0], skip_special_tokens=True)

        qa_pairs = generate_qa(truncated_text)
        formatted_qa.extend(process_qa_output(qa_pairs))

        if (idx + 1) % batch_size == 0:
            save_progress(formatted_qa, output_file)
            print(f"Saved progress at Tafsir entry {idx + 1}")

    save_progress(formatted_qa, output_file)
    print("Tafsir QA generation completed.")


def generate_qa_from_hadith_maarif():
    input_file = "tafsir_data_maarif-ul-quran.json"
    output_file = "maarif-ul-quran_instruction_test.json"

    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as outfile:
            try:
                formatted_qa = json.load(outfile)
            except json.JSONDecodeError:
                formatted_qa = []
    else:
        formatted_qa = []

    with open(input_file, "r", encoding="utf-8") as file:
        hadith_data = json.load(file)

    batch_size = 10
    for idx, hadith in enumerate(hadith_data):
        text = f"{hadith['Narrator']} said: {hadith['tafsir_text']}"
        qa_pairs = generate_qa(text)
        formatted_qa.extend(process_qa_output(qa_pairs))

        if (idx + 1) % batch_size == 0:
            save_progress(formatted_qa, output_file)
            print(f"Saved progress at Hadith entry {idx + 1}")

    save_progress(formatted_qa, output_file)
    print("Hadith QA generation completed.")

# -------------------- 3. Generate QA from Quran --------------------
def generate_qa_from_quran():
    input_file = "quran_data.json"
    output_file = "quran_instruction_test.json"

    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as outfile:
            try:
                formatted_qa = json.load(outfile)
            except json.JSONDecodeError:
                formatted_qa = []
    else:
        formatted_qa = []

    with open(input_file, "r", encoding="utf-8") as file:
        quran_data = json.load(file)

    batch_size = 100
    merged_verses = {}
    
    # Merge 5-10 consecutive verses from the same chapter
    for verse in quran_data:
        chapter_id = verse["chapter_id"]
        if chapter_id not in merged_verses:
            merged_verses[chapter_id] = []
        merged_verses[chapter_id].append(verse["translation"])
    
    merged_texts = []
    for chapter_id, verses in merged_verses.items():
        for i in range(0, len(verses), 10):  # Merge 5-10 verses at a time
            merged_texts.append(" ".join(verses[i:i+10]))

    for idx, merged_text in enumerate(merged_texts):
        qa_pairs = generate_qa(merged_text)
        formatted_qa.extend(process_qa_output(qa_pairs))

        if (idx + 1) % batch_size == 0:
            save_progress(formatted_qa, output_file)
            print(f"Saved progress at Quran entry {idx + 1}")

    save_progress(formatted_qa, output_file)
    print("Quran QA generation completed.")

# -------------------- RUN FUNCTIONS --------------------
if __name__ == "__main__":
    # generate_qa_from_tafsir()
    # generate_qa_from_hadith()
    # generate_qa_from_quran()
    generate_qa_from_tafsir_maarif()
