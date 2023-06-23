from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    LlamaTokenizer,
    LlamaForCausalLM,
)

import torch

# def apply_delta(base_model_path, target_model_path, delta_path):
#     print(f"Loading the delta weights from {delta_path}")
#     delta_tokenizer = AutoTokenizer.from_pretrained(delta_path, use_fast=False)
#     delta = AutoModelForCausalLM.from_pretrained(
#         delta_path, torch_dtype=torch.float16, low_cpu_mem_usage=True
#     )

#     print(f"Loading the base model from {base_model_path}")
#     base = AutoModelForCausalLM.from_pretrained(
#         base_model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True
#     )

#     print("Applying the delta")
#     for name, param in tqdm(delta.state_dict().items(), desc="Applying delta"):
#         assert name in base.state_dict()
#         param.data += base.state_dict()[name]

        
if __name__ == "__main__":
    delta_path = "/home/jovyan/llm-group9/shijh/BayLing-7B"
    model = LlamaForCausalLM.from_pretrained(delta_path, torch_dtype=torch.float16, low_cpu_mem_usage=True)
        #low_cpu_mem_usage=True,
    base_model_path="/home/jovyan/llm-course/pretrained_models/llama-7b-hf/" 
    target_model_path="/home/jovyan/llm-group9/shijh/BayLing-7B/"
    #delta_path="/home/jovyan/llm-course/pretrained_models/bayling-7b-diff/"