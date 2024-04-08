from transformers import AutoTokenizer, GemmaForCausalLM
import torch
import time

def inference(input_text):
    start_time = time.time()
    input_ids = tokenizer(input_text, return_tensors="pt").to(model.device)
    input_length = input_ids["input_ids"].shape[1]
    outputs = model.generate(
        input_ids=input_ids["input_ids"], 
        max_length=1024,
        do_sample=False)
    generated_sequence = outputs[:, input_length:].tolist()
    res = tokenizer.decode(generated_sequence[0])
    end_time = time.time()
    return {"output": res, "latency": end_time - start_time}

model_id = "NexaAIDev/Octopus-v2"
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Check if Metal device is available
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    raise ValueError("Metal device not found. This code requires a compatible Apple device with Metal support.")

try:
    model = GemmaForCausalLM.from_pretrained(model_id)
    model.to(device)
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

input_text = "Take a selfie for me with front camera"

# Validate input text
max_length = 1000
if not input_text.strip():
    raise ValueError("Input text cannot be empty.")
if len(input_text) > max_length:
    raise ValueError(f"Input text cannot exceed {max_length} characters.")

nexa_query = f"""Below is a query from the user. Please respond with the name of the function to call and the parameters to pass to it, formatted as a Python dictionary like:
{{
  "function": "function_name",
  "parameters": {{
    "param1": "value1",
    "param2": "value2"
  }}  
}}

User Query: {input_text}

Response:
"""

start_time = time.time()
try:
    result = inference(nexa_query)
    print("Nexa model result:\n", result["output"])
    print("Latency:", result["latency"], "s")
except Exception as e:
    print(f"Error during inference: {e}")