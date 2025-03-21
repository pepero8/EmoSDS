#!/bin/bash

METAROOT="path/to/your/stage2/model"
DATAROOT="data/unified"
OUTROOT="./"
CACHEROOT="${DATAROOT}/cache/"


mkdir -p ${CACHEROOT}/tokenized/train/
mkdir -p ${CACHEROOT}/tokenized/valid/

echo "================================= stage3 fine-tuning ================================="

export NCCL_P2P_DISABLE=1
torchrun \
    --nproc_per_node 2 \
    --standalone \
src/train/sft_residual.py \
    --train_task "unified" \
    --emo_token_list "<anger> <happiness> <neutral> <sadness> <surprise>" \
    --model_name_or_path "${METAROOT}" \
    --data_path "${DATAROOT}/unified_task_esd_train.jsonl" \
    --val_data_path "${DATAROOT}/unified_task_esd_valid.jsonl" \
    --test_data_path "${DATAROOT}/unified_task_esd_test.jsonl" \
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
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --eval_accumulation_steps 10 \
    --num_train_epochs 5 \
    --eval_strategy "steps" \
    --eval_steps 231 \
    --save_strategy "steps" \
    --save_steps 231 \
    --load_best_model_at_end True \
    --metric_for_best_model "bleu_res_text" \
    --greater_is_better True \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.05 \
    --lr_scheduler_type "cosine" \
    --log_level debug \
    --logging_steps 1 \
    --overwrite_output_dir
