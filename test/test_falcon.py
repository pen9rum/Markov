import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "tiiuae/Falcon-H1-7B-Instruct"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)
model = model.to("cuda")
model.eval()

messages = [{"role": "user", "content": "hello"}]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

print("Tokenizing...")
inputs = tokenizer([text], return_tensors="pt", padding=True, truncation=True)
inputs = {k: v.to("cuda") for k, v in inputs.items()}

print("Generating...")
with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=16,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

gen_ids = output_ids[0][inputs["input_ids"].shape[1]:]
print(tokenizer.decode(gen_ids, skip_special_tokens=True))