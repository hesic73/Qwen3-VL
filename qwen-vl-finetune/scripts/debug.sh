#!/bin/bash

export NPROC_PER_NODE=1

deepspeed=./scripts/zero3_offload.json

llm="Qwen/Qwen3-VL-4B-Instruct"

datasets="/home/hsc/26spring/wave_sheet_real/data/my_qwen_dataset/annotations.jsonl"
output_dir="./output/qwen3vl_lora_run"


torchrun --nproc_per_node=${NPROC_PER_NODE} \
    qwenvl/train/train_qwen.py \
    --deepspeed ${deepspeed} \
    --model_name_or_path "${llm}" \
    --dataset_use ${datasets} \
    --data_flatten True \
    --lora_enable True \
    --bf16 \
    --output_dir ${output_dir} \
    --num_train_epochs 3 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --eval_strategy "no" \
    --save_strategy "epoch" \
    --save_total_limit 2 \
    --learning_rate 5e-5 \
    --weight_decay 0 \
    --max_grad_norm 1 \
    --lr_scheduler_type "linear" \
    --logging_steps 1 \
    --model_max_length 1024 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --run_name "qwen3vl-lora-rl-data" \
    --report_to "wandb"
