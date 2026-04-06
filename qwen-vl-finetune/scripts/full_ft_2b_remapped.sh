#!/bin/bash
# Full fine-tuning script for Qwen3-VL-2B using remapped <ACT_i> / <AREA_i> tokens.

export NPROC_PER_NODE=1

GPU_ID=$(bash scripts/find_gpu.sh)
echo "Using GPU: ${GPU_ID}"
export CUDA_VISIBLE_DEVICES=${GPU_ID}

deepspeed=./scripts/zero2.json

llm="Qwen/Qwen3-VL-2B-Instruct"
custom_tokenizer="./custom_tokenizer"
datasets="/data2/sichenghe/26spring/my_qwen_dataset/annotations.jsonl"
output_dir="./output/qwen3vl_2b_full_ft_remapped"


torchrun --nproc_per_node=${NPROC_PER_NODE} \
    qwenvl/train/train_qwen.py \
    --deepspeed ${deepspeed} \
    --model_name_or_path "${llm}" \
    --tokenizer_path "${custom_tokenizer}" \
    --dataset_use ${datasets} \
    --data_flatten True \
    --tune_mm_llm True \
    --tune_mm_mlp True \
    --tune_mm_vision False \
    --bf16 \
    --output_dir ${output_dir} \
    --num_train_epochs 3 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --eval_strategy "no" \
    --save_strategy "epoch" \
    --save_total_limit 2 \
    --learning_rate 1e-5 \
    --weight_decay 0 \
    --max_grad_norm 1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 1024 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --run_name "qwen3vl-2b-full-ft-remapped-tokens" \
    --report_to "wandb"
