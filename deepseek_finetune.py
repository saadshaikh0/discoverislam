import torch
import os
import json
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    Trainer
)
from peft import get_peft_model, LoraConfig, TaskType

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

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    max_memory={0: "7GB"},
    quantization_config=bnb_config,
    trust_remote_code=True,
    low_cpu_mem_usage=True
)

# OPTIONAL: For lower VRAM usage
# model.config.use_cache = False
# model.gradient_checkpointing_enable()

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
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

qa_files = ["hadith_instruction_test.json", "tafsir_instruction_test.json","tafsir_maarif_instruction_test.json", "quran_instruction_test.json"]
qa_dataset = load_qa_data(qa_files)

def format_qa(example):
    # You might want to incorporate any special Qwen instructions or style:
    # https://github.com/QwenLM/Qwen-7B/blob/main/README.md
    # or whatever prompt format is recommended
    user_prompt = f"User: {example['instruction']}\nAI:"
    answer = example["output"]
    return {
        "full_text": user_prompt + answer
    }

hf_dataset = Dataset.from_list([format_qa(q) for q in qa_dataset])

def tokenize_function(examples):
    # We only have "full_text" which is prompt+answer concatenated
    # We'll pad/truncate to a max length, then *later* mask out prompt tokens.
    tokens = tokenizer(
        examples["full_text"],
        padding="max_length",
        truncation=True,
        max_length=512
    )
    return tokens

tokenized_dataset = hf_dataset.map(tokenize_function, batched=True)

# If you want a data collator that automatically sets
# 15% of tokens to masked or anything, use DataCollatorForLanguageModeling
# but typically for SFT, we do a "prompt masking" approach, so you might
# want a custom collator that sets the prompt portion to -100.
# For simplicity, here's a custom one:
def causal_lm_data_collator(features):
    batch = {}
    # Convert list of dicts => dict of lists => tensor
    input_ids = torch.tensor([f["input_ids"] for f in features], dtype=torch.long)
    attention_mask = torch.tensor([f["attention_mask"] for f in features], dtype=torch.long)

    # By default, train on the entire sequence, but if you'd like to ignore
    # the prompt portion, you can set it to -100. This requires you to know
    # where prompt ends and answer begins. If your "prompt + answer" is all
    # we have, we *might* guess a separator. This is a demonstration:
    labels = input_ids.clone()
    # Example: find index of "AI:" or something similar. If you have a simpler
    # known prompt length, you can do a direct slice. 
    # (Leaving it naive here.)

    batch["input_ids"] = input_ids
    batch["labels"] = labels
    batch["attention_mask"] = attention_mask
    return batch

training_args = TrainingArguments(
    output_dir="./fine_tuned_deepseek",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    save_total_limit=2,
    save_steps=500,
    num_train_epochs=3,
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
    tokenizer=tokenizer,
    data_collator=causal_lm_data_collator
)

trainer.train()
model.save_pretrained("./fine_tuned_deepseek")
print("✅ Fine-tuning complete. LoRA weights saved at './fine_tuned_deepseek'")
