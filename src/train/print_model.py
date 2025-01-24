from transformers import AutoModelForCausalLM

# import torch


model = AutoModelForCausalLM.from_pretrained(
    "/home/jhwan98/EmoSDS/SpeechGPT/speechgpt/llama/3_2/3B/Llama-3.2-3B-Instruct",
)

print(type(model))
print(model)
