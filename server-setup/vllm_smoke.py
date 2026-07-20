import os
from vllm import LLM, SamplingParams
llm = LLM(model=os.environ["MODEL_PATH"], dtype="float16", tensor_parallel_size=2,
          max_model_len=2048, gpu_memory_utilization=0.90, enforce_eager=True, trust_remote_code=True)
out = llm.generate(["What is 12 times 8? Answer briefly."], SamplingParams(max_tokens=64, temperature=0))
print("VLLM_OUTPUT:", out[0].outputs[0].text)
print("VLLM_SMOKE_OK")
