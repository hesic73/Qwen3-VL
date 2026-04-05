#!/bin/bash
# Training script using remapped <ACT_i> / <AREA_i> tokens.
#
# Prerequisites (run once):
#   python tools/remap_tokenizer.py \
#       --base_model Qwen/Qwen3-VL-4B-Instruct \
#       --output_dir ./custom_tokenizer
#
# Then re-export your dataset (annotations.jsonl will use <ACT_i>/<AREA_i> directly).

export NPROC_PER_NODE=1

GPU_ID=$(bash scripts/find_gpu.sh)
echo "Using GPU: ${GPU_ID}"
export CUDA_VISIBLE_DEVICES=${GPU_ID}

deepspeed=./scripts/zero3_offload.json

llm="Qwen/Qwen3-VL-4B-Instruct"
custom_tokenizer="./custom_tokenizer"
datasets="/data2/sichenghe/26spring/my_qwen_dataset/annotations.jsonl"
output_dir="./output/qwen3vl_lora_remapped"


torchrun --nproc_per_node=${NPROC_PER_NODE} \
    qwenvl/train/train_qwen.py \
    --deepspeed ${deepspeed} \
    --model_name_or_path "${llm}" \
    --tokenizer_path "${custom_tokenizer}" \
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
    --ddp_find_unused_parameters False \
    --model_max_length 1024 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --run_name "qwen3vl-lora-remapped-tokens" \
    --report_to "wandb"
