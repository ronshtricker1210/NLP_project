import os, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
MODEL = os.environ["MODEL_PATH"]   # node-local copy
question = open("/vol/scratch/%s/scripts/question.txt" % os.environ["USER"]).read().strip()
print("loading from:", MODEL, flush=True)
print("torch:", torch.__version__, "| cuda:", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU", flush=True)
print("QUESTION:", question, flush=True)
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="auto")
msgs = [{"role": "user", "content": question}]
inputs = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                 return_tensors="pt", return_dict=True).to(model.device)
out = model.generate(**inputs, max_new_tokens=4096, do_sample=True, temperature=0.6, top_p=0.95)
text = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print("\n========== MODEL REASONING + ANSWER ==========\n", flush=True)
print(text, flush=True)
print("\n========== END ==========", flush=True)
