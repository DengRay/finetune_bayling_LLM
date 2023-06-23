#! /bin/bash

python generate.py \
    --base_model '/home/jovyan/llm-group9/shijh/BayLing-7B' \
    --prompt_template 'bayling' \
    --lora_weights '/home/jovyan/llm-group9/shijh/alpaca-lora-main/lora-bayling3' \
    #--prompt_template 'bayling' \