#!/bin/bash

METAROOT="path/to/your/stage1/model"
DATAROOT="data/asr_ser"
OUTROOT="./"
CACHEROOT="${DATAROOT}/cache/"


mkdir -p ${CACHEROOT}/tokenized/train/
mkdir -p ${CACHEROOT}/tokenized/valid/

echo "================================= asr+ser fine-tuning ================================="

export CUDA_VISIBLE_DEVICES=0,2
torchrun \
    --nproc_per_node 2 \
    --standalone \
src/train/sft_residual.py \
    --train_task "asr+ser" \
    --emo_token_list "<anger> <happiness> <neutral> <sadness> <surprise>" \
    --model_name_or_path "${METAROOT}" \
    --data_path "${DATAROOT}/asr_ser_task_train_esd.jsonl" \
    --val_data_path "${DATAROOT}/asr_ser_task_valid_esd.jsonl" \
    --test_data_path "${DATAROOT}/asr_ser_task_test_esd.jsonl" \
    --val_set_size 0 \
    --cache_dir ${CACHEROOT} \
    --preprocessing_num_workers 10 \
    --model_max_length 3072 \
    --bf16 True \
    --do_train \
    --do_eval \
    --do_predict \
    --train_on_inputs True \
    --output_dir "${OUTROOT}" \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --eval_accumulation_steps 10 \
    --num_train_epochs 3 \
    --eval_strategy "steps" \
    --eval_steps 55 \
    --save_strategy "steps" \
    --save_steps 55 \
    --load_best_model_at_end True \
    --metric_for_best_model "loss" \
    --greater_is_better False \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.1 \
    --lr_scheduler_type "cosine" \
    --log_level debug \
    --overwrite_output_dir \
    --train_low_layers \
    --logging_steps 1
