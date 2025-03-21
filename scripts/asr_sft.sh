#!/bin/bash

METAROOT="llama/3_2/3B/Llama-3.2-3B-Instruct"
DATAROOT="data/asr"
OUTROOT="./"
CACHEROOT="${DATAROOT}/cache/"


mkdir -p ${CACHEROOT}/tokenized/train/
mkdir -p ${CACHEROOT}/tokenized/valid/

echo "================================= asr fine-tuning ================================="

export NCCL_P2P_DISABLE=1
torchrun \
    --nproc_per_node 2 \
    --standalone \
src/train/sft.py \
    --train_task "asr" \
    --model_name_or_path "${METAROOT}" \
    --data_path "${DATAROOT}/asr_task_librispeech.jsonl" \
    --val_set_size 100 \
    --cache_dir ${CACHEROOT} \
    --preprocessing_num_workers 10 \
    --model_max_length 2048 \
    --bf16 True \
    --do_train \
    --do_eval \
    --train_on_inputs True \
    --output_dir "${OUTROOT}" \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --eval_accumulation_steps 10 \
    --num_train_epochs 10 \
    --eval_strategy "steps" \
    --eval_steps 223 \
    --save_strategy "steps" \
    --save_steps 223 \
    --load_best_model_at_end True \
    --metric_for_best_model "wer" \
    --greater_is_better False \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.1 \
    --lr_scheduler_type "cosine" \
    --log_level debug \
    --overwrite_output_dir \
    --train_low_layers \
    --logging_steps 1 \
