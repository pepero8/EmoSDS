#!/bin/bash

METAROOT="/home/jhwan98/EmoSDS/SpeechGPT/speechgpt/llama/3_2/3B/Llama-3.2-3B-Instruct"
# METAROOT="/home/jhwan98/EmoSDS/SpeechGPT/speechgpt/output/stage2"
DATAROOT="data/asr_ser/layer7_k2000"
OUTROOT="/shared/NAS_SSD/jhl/futureinternet/output/asr_ser_7layer_k2000"
CACHEROOT="${DATAROOT}/cache/"


mkdir -p ${CACHEROOT}/tokenized/train/
mkdir -p ${CACHEROOT}/tokenized/valid/

echo "================================= asr+ser fine-tuning ================================="


# --nproc_per_node를 사용할 gpu 개수로 설정하면 됨
export CUDA_VISIBLE_DEVICES=1,2
torchrun \
    --nproc_per_node 2 \
    --standalone \
src/train/sft.py \
    --train_task "asr+ser" \
    --style_token_list "<anger> <disgust> <fear> <happiness> <neutral> <sadness> <surprise>" \
    --model_name_or_path "${METAROOT}" \
    --data_path "${DATAROOT}/asr_ser_task_train_balanced.jsonl" \
    --val_data_path "${DATAROOT}/asr_ser_task_valid_balanced.jsonl" \
    --test_data_path "${DATAROOT}/asr_ser_task_test_balanced.jsonl" \
    --val_set_size 0 \
    --cache_dir ${CACHEROOT} \
    --preprocessing_num_workers 10 \
    --model_max_length 1024 \
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
    --num_train_epochs 10 \
    --eval_strategy "steps" \
    --eval_steps 89 \
    --save_strategy "steps" \
    --save_steps 89 \
    --load_best_model_at_end True \
    --greater_is_better False \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.1 \
    --lr_scheduler_type "cosine" \
    --log_level debug \
    --overwrite_output_dir \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \

