"""Generate text with a locally fine-tuned model."""

import sys

from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = sys.argv[1] if len(sys.argv) > 1 else "trained_models/hpc_demo/v1"
PROMPT = sys.argv[2] if len(sys.argv) > 2 else "Once upon a time"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)

inputs = tokenizer(PROMPT, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=30, do_sample=True, temperature=0.8)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
