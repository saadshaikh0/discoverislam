import torch
import os
import json
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer
)
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    prepare_model_for_kbit_training
)

model_name = "Qwen/Qwen2.5-3B-Instruct"

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.backends.cuda.matmul.allow_tf32 = True
torch.cuda.empty_cache()

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

bnb_config = {
    "load_in_4bit": True,
    "bnb_4bit_use_double_quant": True,
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_compute_dtype": torch.bfloat16,
}

print("Loading model in 4-bit...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    quantization_config=bnb_config,
    trust_remote_code=True
)

# Important: Prepare for k-bit training
model = prepare_model_for_kbit_training(model)

# (Optional) Turn off cache if needed
model.config.use_cache = False
model.gradient_checkpointing_enable()

# Check how attention modules are named
# for name, param in model.named_parameters():
#     print(name)
#     break

# Suppose QWen's attention submodules are named "q_proj", "k_proj", "v_proj", etc.
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    # Make sure these match actual submodule names in Qwen
    target_modules=["q_proj", "k_proj", "v_proj"]  
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

def load_qa_data(filenames):
    qa_data = []
    for file in filenames:
        with open(file, "r", encoding="utf-8") as f:
            qa_data.extend(json.load(f))
    return qa_data

task_files = [
    "hadith_instruction_test.json",
    "tafsir_instruction_test.json",
    "tafsir_maarif_instruction_test.json",
    "quran_instruction_test.json"
]
dataset = load_qa_data(task_files)

def format_qa(example):
    messages = [
        {"role": "user", "content": example['instruction']},
        {"role": "assistant", "content": example['output']}
    ]
    formatted_input = tokenizer.apply_chat_template(messages, tokenize=False)
    return {"full_text": formatted_input}

dataset_formatted = Dataset.from_list([format_qa(q) for q in dataset])

def tokenize_function(examples):
    return tokenizer(
        examples["full_text"],
        padding="max_length",
        truncation=True,
        max_length=512
    )

tokenized_dataset = dataset_formatted.map(tokenize_function, batched=True)

def causal_lm_data_collator(features):
    input_ids = torch.tensor([f["input_ids"] for f in features], dtype=torch.long)
    attention_mask = torch.tensor([f["attention_mask"] for f in features], dtype=torch.long)
    labels = input_ids.clone()
    
    for i in range(input_ids.shape[0]):
        full_text = tokenizer.decode(input_ids[i], skip_special_tokens=True)
        ai_index = full_text.find("<|im_start|>assistant")
        if ai_index != -1:
            tokens_until_ai = tokenizer(full_text[:ai_index], add_special_tokens=False)["input_ids"]
            prompt_length = len(tokens_until_ai)
            labels[i, :prompt_length] = -100
        else:
            labels[i, :] = -100
    
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

training_args = TrainingArguments(
    output_dir="./fine_tuned_qwen",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    save_total_limit=2,
    save_steps=500,
    num_train_epochs=1,
    learning_rate=2e-5,
    logging_dir="./logs",
    logging_steps=10,
    bf16=True,
    optim="adamw_torch"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=causal_lm_data_collator
)

trainer.train()
model.save_pretrained("./fine_tuned_qwen")
print("✅ Fine-tuning complete. LoRA weights saved at './fine_tuned_qwen'")
