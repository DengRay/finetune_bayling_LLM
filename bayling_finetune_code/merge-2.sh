#! /bin/bash

python export_hf_checkpoint.py \
    --base_model ../BayLing-7B/ \
    --lora_model lora-bayling3/ \
    --output_model lora-bayling3/hf