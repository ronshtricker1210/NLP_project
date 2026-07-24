"""Minimal one-request test to confirm the token + router + model work.
API twin of gcp-setup/vllm_smoke.py — costs a fraction of a cent.
    source api-setup/env.sh && python api_smoke.py
"""
import os
from openai import OpenAI

client = OpenAI(api_key=os.environ["HF_TOKEN"],
                base_url=os.environ.get("API_BASE", "https://router.huggingface.co/v1"))
resp = client.chat.completions.create(
    model=os.environ.get("MODEL", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B:nscale"),
    messages=[{"role": "user", "content": "What is 12 times 8? Answer briefly."}],
    max_tokens=256, temperature=0.6, timeout=120)
print("API_OUTPUT:", resp.choices[0].message.content)
print("usage:", resp.usage.prompt_tokens, "in /", resp.usage.completion_tokens, "out")
print("API_SMOKE_OK")
