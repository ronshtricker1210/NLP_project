"""Minimal vLLM load + one-prompt test to confirm the GPU + install work.
GCP / single-L4: bfloat16, tensor_parallel_size=1.
Honors VLLM_ENFORCE_EAGER=1 (set by env.sh) to skip runtime JIT on bare VMs.
    source gcp-setup/env.sh && python vllm_smoke.py
"""
import os
from vllm import LLM, SamplingParams

model = os.environ.get("MODEL_PATH", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
eager = os.environ.get("VLLM_ENFORCE_EAGER", "") == "1"
llm = LLM(model=model, dtype="bfloat16", tensor_parallel_size=1,
          max_model_len=2048, gpu_memory_utilization=0.90,
          enforce_eager=eager, trust_remote_code=True)
out = llm.generate(["What is 12 times 8? Answer briefly."],
                   SamplingParams(max_tokens=64, temperature=0))
print("VLLM_OUTPUT:", out[0].outputs[0].text)
print("VLLM_SMOKE_OK")
