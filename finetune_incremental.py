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
from peft import PeftModel, get_peft_model, LoraConfig, TaskType

# ✅ Load Previously Fine-Tuned Model
model_path = "./fine_tuned_deepseek/checkpoint-2884"  # Update to latest checkpoint

# ✅ Optimize CUDA Memory Usage
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ✅ Enable 4-bit Quantization (Reduces VRAM)
bnb_config = {
    "load_in_4bit": True,
    "bnb_4bit_use_double_quant": True,
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_compute_dtype": torch.bfloat16,  # ✅ Optimized for Ampere+ GPUs
}

torch.backends.cuda.matmul.allow_tf32 = True
torch.cuda.empty_cache()

# ✅ Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# ✅ Load Model with Auto Offloading to Reduce GPU Usage
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",  # ✅ Automatically offloads to CPU/GPU
    max_memory={0: "6GB"},  # ✅ Restrict memory usage
    quantization_config=bnb_config,
    trust_remote_code=True,
    low_cpu_mem_usage=True
)

# ✅ Enable Gradient Checkpointing to Reduce VRAM
model.gradient_checkpointing_enable()

# ✅ Load LoRA Adapter (to continue fine-tuning)
model = PeftModel.from_pretrained(model, model_path)

# ✅ Ensure LoRA Layers Are Trainable
model.train()
for name, param in model.named_parameters():
    if param.requires_grad:
        param.requires_grad = True  # ✅ Only enable gradient calculation for trainable layers

# ✅ Load New QA Dataset
def load_qa_data(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)

qa_files = ["tafsir_maarif_instruction_test.json"]
qa_dataset = load_qa_data(qa_files[0])

# ✅ Convert to Hugging Face Dataset Format
def format_qa(example):
    return {
        "instruction": f"User: {example['instruction']}\nAI:",
        "output": example["output"]
    }

hf_dataset = Dataset.from_list([format_qa(q) for q in qa_dataset])

# ✅ Tokenization Function (Reduce `max_length` to Save Memory)
def tokenize_function(examples):
    return tokenizer(
        examples["instruction"],
        text_target=examples["output"],
        padding="max_length",
        truncation=True,
        max_length=1024  # ✅ Reduce length to save memory
    )

tokenized_dataset = hf_dataset.map(tokenize_function, batched=True)

# ✅ Adjust Training Parameters for Incremental Fine-Tuning
training_args = TrainingArguments(
    output_dir="./fine_tuned_deepseek_maarif",
    per_device_train_batch_size=1,  # ✅ Small batch to fit in 8GB VRAM
    gradient_accumulation_steps=8,  # ✅ Reduce memory usage
    save_total_limit=2,
    save_steps=500,
    num_train_epochs=3,  # ✅ Reduce epochs for incremental fine-tuning
    learning_rate=2e-5,
    logging_dir="./logs",
    logging_steps=50,
    bf16=True,  # ✅ Use `bf16` instead of `fp16` to prevent scaling issues
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

model = model.merge_and_unload()

# ✅ Save the Updated Model
trainer.save_model("./fine_tuned_deepseek_maarif")

print("✅ Fine-tuning complete. Model saved at './fine_tuned_deepseek_maarif'")
