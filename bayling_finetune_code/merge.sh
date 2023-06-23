#! /bin/bash

base_model_path="/home/jovyan/llm-course/pretrained_models/llama-7b-hf/" 
target_model_path="/home/jovyan/llm-group9/shijh/BayLing-7B/"
delta_path="/home/jovyan/llm-course/pretrained_models/bayling-7b-diff/"

python BayLing-main/apply_delta.py \
    --base-model-path "${base_model_path}" \
    --target-model-path "${target_model_path}" \
    --delta-path "${delta_path}"