from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
import json
import re

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

# Load the Tafsir JSON file
with open("tafsir_data.json", "r", encoding="utf-8") as file:
    tafsir_data = json.load(file)

# Select the first tafsir entry for testing
test_tafsir = tafsir_data[25]

# Clean tafsir text
cleaned_text = remove_html_tags(test_tafsir["tafsir_text"])

# Truncate text if it exceeds max_length
max_length = 5000
encoded_text = tokenizer.encode(cleaned_text, truncation=True, max_length=max_length, return_tensors="pt")
truncated_text = tokenizer.decode(encoded_text[0], skip_special_tokens=True)

# Function to generate question-answer pairs
def generate_qa(text):
    prompt = f"Generate three question-answer pairs based on the following Islamic tafsir:\n\n{text}\n\nQuestions and Answers:"
    response = qa_pipeline(prompt, max_new_tokens=1024, do_sample=True, temperature=0.7)
    return response[0]["generated_text"]

# Generate QA pairs
qa_pairs = generate_qa(truncated_text)

# Reformat QA pairs to ensure proper structure
qa_list = qa_pairs.split("\n")
formatted_qa = []
current_question = None
current_answer = ""

for line in qa_list:
    line = line.strip()

    # Skip empty lines and irrelevant headers
    if not line or line.lower() in ["questions and answers:", ""]:
        continue
    
    # Detect questions
    if line.lower().startswith("question:") or re.match(r"^\d+\.", line):
        # Save previous QA pair before starting a new one
        if current_question and current_answer:
            formatted_qa.append({"instruction": current_question, "output": current_answer.strip()})
        
        # Extract the question (handling cases with numbering like '1. Question: ...')
        current_question = re.sub(r"^\d+\.\s?", "", line.replace("Question:", "").strip())
        current_answer = ""  # Reset answer
    
    # Detect answers
    elif line.lower().startswith("answer:"):
        current_answer = line.replace("Answer:", "").strip()
    
    # Handle multi-line answers (if the next line is not a new question)
    elif current_question and current_answer:
        current_answer += " " + line  # Append to the existing answer

# Add last QA pair
if current_question and current_answer:
    formatted_qa.append({"instruction": current_question, "output": current_answer.strip()})

# Save the result in a JSON file
with open("tafsir_instruction_test.json", "w", encoding="utf-8") as outfile:
    json.dump(formatted_qa, outfile, indent=4, ensure_ascii=False)

print("Tafsir question-answer pairs for one entry have been successfully saved.")
