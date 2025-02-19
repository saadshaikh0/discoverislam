import torch
import json
import os
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    Trainer
)
from peft import get_peft_model, LoraConfig, TaskType

# ✅ Use 4B Instead of 7B (Better for 8GB VRAM)
model_name = "Qwen/Qwen2.5-3B-Instruct"
# ✅ Set CUDA Memory Configurations
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ✅ Enable 4-bit Quantization (Reduces VRAM)
bnb_config = {
    "load_in_4bit": True,
    "bnb_4bit_use_double_quant": True,
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_compute_dtype": torch.bfloat16,  # ✅ Saves memory
}

torch.backends.cuda.matmul.allow_tf32 = True
torch.cuda.empty_cache()  # ✅ Clears unused memory

# ✅ Load Tokenizer & Model (No CPU Offloading)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

tokenizer.pad_token = tokenizer.eos_token  # Use EOS token as pad token
tokenizer.padding_side = "right"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",  # ✅ Auto-allocates layers to GPU
    max_memory={0: "7GB"},  # ✅ Restricts GPU Memory Usage
    quantization_config=bnb_config,
    trust_remote_code=True,
    low_cpu_mem_usage=True
)

# ✅ Apply LoRA for Efficient Fine-Tuning
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=2,  # ✅ Lower rank to reduce VRAM usage
    lora_alpha=4,  # ✅ Reduce parameter scaling
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj"]
)
model = get_peft_model(model, peft_config)

# ✅ Load QA Datasets
def load_qa_data(filenames):
    qa_data = []
    for file in filenames:
        with open(file, "r", encoding="utf-8") as f:
            qa_data.extend(json.load(f))
    return qa_data

qa_files = ["hadith_instruction_test.json", "tafsir_instruction_test.json", "quran_instruction_test.json"]
qa_dataset = load_qa_data(qa_files)  # ✅ Start with 500 samples for testing

# ✅ Convert to Hugging Face Dataset Format
def format_qa(example):
    return {
        "instruction": f"User: {example['instruction']}\nAI:",
        "output": example["output"]
    }

hf_dataset = Dataset.from_list([format_qa(q) for q in qa_dataset])

# ✅ Tokenization Function
def tokenize_function(examples):
    return tokenizer(
        examples["instruction"],
        text_target=examples["output"],
        padding="max_length",
        truncation=True,
        max_length=512
    )

tokenized_dataset = hf_dataset.map(tokenize_function, batched=True)

# ✅ Reduce Batch Size & Steps to Prevent OOM
training_args = TrainingArguments(
    output_dir="./fine_tuned_deepseek",
    per_device_train_batch_size=1,  # ✅ Small batch to fit in 8GB VRAM
    gradient_accumulation_steps=8,  # ✅ Accumulates gradients to simulate bigger batch
    save_total_limit=1,
    save_steps=1000,
    num_train_epochs=2,
    learning_rate=3e-5,  # ✅ Lower LR for stability
    logging_dir="./logs",
    logging_steps=50,
    fp16=True,  # ✅ Mixed precision
    optim="adamw_torch"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorForSeq2Seq(tokenizer)
)

# ✅ Start Training
trainer.train()

# ✅ Save Fine-Tuned Model
trainer.save_model("./fine_tuned_deepseek")

print("✅ Fine-tuning complete. Model saved at './fine_tuned_deepseek'")
