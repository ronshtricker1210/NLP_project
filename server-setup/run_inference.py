"""Minimal smoke test: load DeepSeek-R1-Distill-Qwen-7B and run one prompt.
Confirms the download + GPU + offline cache all work before scaling up.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

print("Loading tokenizer...")
tok = AutoTokenizer.from_pretrained(MODEL)
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.bfloat16, device_map="auto"
)

msgs = [{"role": "user", "content": "What is 12 * 8? Think step by step."}]
inputs = tok.apply_chat_template(
    msgs, add_generation_prompt=True, return_tensors="pt"
).to(model.device)

print("Generating...")
out = model.generate(inputs, max_new_tokens=512, do_sample=False)
print(tok.decode(out[0][inputs.shape[1]:], skip_special_tokens=True))
