import os
import sys
import json
import random

import fire
import gradio as gr
import torch
import transformers
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
from torchsummary import summary

from utils.callbacks import Iteratorize, Stream
from utils.prompter import Prompter

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
    


def main(
    load_8bit: bool = False,
    base_model: str = "/home/jovyan/llm-group9/shijh/BayLing-7B",
    lora_weights: str = "/home/jovyan/llm-group9/shijh/alpaca-lora-main/lora-bayling2",
    prompt_template: str = "",  # The prompt template to use, will default to alpaca.
):
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"

    prompter = Prompter(prompt_template)
    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    if device == "cuda":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        if lora_weights != "":
            model = PeftModel.from_pretrained(
                model,
                lora_weights,
                torch_dtype=torch.float16,
            )
    else:
        print("gg")

    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    if not load_8bit:
        model.half()  # seems to fix bugs for some users.
    
    print(model)
    
    # print(f"{*} * 20")        
        

if __name__ == "__main__":
    fire.Fire(main)
        