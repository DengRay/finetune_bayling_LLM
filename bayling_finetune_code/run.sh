#! /bin/bash

#base_model="/home/jovyan/llm-group9/shijh/BayLing-7B"
base_model="/home/jovyan/llm-course/pretrained_models/llama-7b-hf"
saved_model="/home/jovyan/llm-group9/shijh/saved_model"
data_path="/home/jovyan/llm-course/instruction_data/alpaca_data.json"

#torchrun --nproc_per_node=1 train.py \
#    --model_name_or_path ${base_model} \
#    --data_path ./alpaca_data.json \
#    --bf16 True \
#    --output_dir ${saved_model} \
#    --num_train_epochs 1 \
#    --per_device_train_batch_size 4 \
#    --per_device_eval_batch_size 4 \
#    --gradient_accumulation_steps 8 \
#    --evaluation_strategy "no" \
#    --save_strategy "steps" \
#    --save_steps 2000 \
#    --save_total_limit 1 \
#    --learning_rate 2e-5 \
#    --weight_decay 0. \
#    --warmup_ratio 0.03 \
#    --lr_scheduler_type "cosine" \
#    --logging_steps 1 \
#    --fsdp "full_shard auto_wrap" \
#    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
#    --tf32 True
    
#torchrun --nproc_per_node=1 train.py \
#    --model_name_or_path ${base_model} \
#    --data_path ${data_path} \
#    --bf16 True \
#    --output_dir ${saved_model} \
#    --num_train_epochs 1 \
#    --per_device_train_batch_size 1 \
#    --per_device_eval_batch_size 1 \
#    --gradient_accumulation_steps 2 \
#    --evaluation_strategy "no" \
#    --save_strategy "steps" \
#    --save_steps 2000 \
#    --save_total_limit 1 \
#    --learning_rate 2e-5 \
#    --weight_decay 0. \
#    --warmup_ratio 0.03 \
#    --lr_scheduler_type "cosine" \
#    --logging_steps 1

python train.py \
    --model_name_or_path ${base_model} \
    --data_path ${data_path} \
    --output_dir ${saved_model} \