#! /bin/bash

python finetune.py \
    --base_model '/home/jovyan/llm-group9/shijh/BayLing-7B' \
    --load_in_8bit True \
    --data_path '/home/jovyan/llm-group9/shijh/alpaca-lora-main/instruction_final.json' \
    --output_dir './lora-bayling3' \
    --batch_size 128 \
    --micro_batch_size 32 \
    --num_epochs 3 \
    --learning_rate 1e-4 \
    --cutoff_len 512 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target_modules '[q_proj,k_proj,v_proj,o_proj]' \
    --train_on_inputs False \
    --prompt_template_name 'bayling' \
    --group_by_length \
    --val_set_size 2000