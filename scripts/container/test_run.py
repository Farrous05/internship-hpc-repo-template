import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    model_id = "tiiuae/falcon-7b-instruct"  # Replace with your desired model
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",  # Enable FlashAttention-2
    ).to(device)

    prompt = "Hello, world! How are you?"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
        )

    print("Generated:", tokenizer.decode(outputs[0], skip_special_tokens=True))

if __name__ == "__main__":
    main()
