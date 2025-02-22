import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ✅ Paths for Models
base_model_name = "Qwen/Qwen2.5-3B-Instruct"  # Change if using a different base model
fine_tuned_model_path = "./fine_tuned_deepseek"  # Change path if needed

# ✅ Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(fine_tuned_model_path, trust_remote_code=True)

# ✅ Ensure Proper Tokenization
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
tokenizer.model_max_length = 2048

# ✅ 4-bit Quantization Config
bnb_config = {
    "load_in_4bit": True,
    "bnb_4bit_use_double_quant": True,
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_compute_dtype": torch.float16,
}

# ✅ Load Base Model
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    quantization_config=bnb_config,
    device_map="cuda",
    trust_remote_code=True
)

# ✅ Load Fine-Tuned Model & Merge LoRA Adapter
fine_tuned_model = AutoModelForCausalLM.from_pretrained(
    fine_tuned_model_path,
    quantization_config=bnb_config,
    device_map="cuda",
    trust_remote_code=True
)
fine_tuned_model = PeftModel.from_pretrained(fine_tuned_model, fine_tuned_model_path)
fine_tuned_model = fine_tuned_model.merge_and_unload()  # ✅ Merge LoRA weights before inference
fine_tuned_model.eval()

# ✅ Function to Generate Response
def generate_response(model, user_input):
    prompt = f"<|im_start|>system\nYou are a helpful AI assistant.\n<|im_end|>\n<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant\n"

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    output_tokens = model.generate(
        **inputs,
        max_new_tokens=500,
        temperature=0.4,  # More deterministic
        top_p=0.85,
        repetition_penalty=1.1,  # Avoid repetition
        do_sample=True
    )

    return tokenizer.decode(output_tokens[0], skip_special_tokens=True)

# ✅ Run Comparison Test
if __name__ == "__main__":
    print("\n🚀 Model Comparison: Base Model vs Fine-Tuned Model")
    while True:
        user_query = input("\n📝 Enter your query (type 'exit' to quit): ")
        if user_query.lower() == "exit":
            break

        print("\n🔍 Generating responses...")

        base_response = generate_response(base_model, user_query)
        fine_tuned_response = generate_response(fine_tuned_model, user_query)

        print("\n🎯 **Base Model Response:**")
        print(base_response)

        print("\n🚀 **Fine-Tuned Model Response:**")
        print(fine_tuned_response)
