#!/bin/bash

# METAROOT="/shared/NAS_SSD/jhl/futureinternet/output/asr/"
# METAROOT="/shared/NAS_SSD/jhl/futureinternet/output/stage3/checkpoint-3600"
METAROOT="/shared/NAS_SSD/jhl/futureinternet/output/asr_ser_6layer/checkpoint-340"
DATAROOT="data/unified/layer6_k1000_downsampled"
OUTROOT="/shared/NAS_SSD/jhl/futureinternet/output/unified_6layer_k1000_downsampled"
CACHEROOT="${DATAROOT}/cache/"


mkdir -p ${CACHEROOT}/tokenized/train/
mkdir -p ${CACHEROOT}/tokenized/valid/

echo "================================= unified fine-tuning ================================="


# --nproc_per_node를 사용할 gpu 개수로 설정하면 됨
export CUDA_VISIBLE_DEVICES=1,2
torchrun \
    --nproc_per_node 2 \
    --standalone \
src/train/sft.py \
    --train_task "unified" \
    --style_token_list "<anger> <disgust> <fear> <happiness> <neutral> <sadness> <surprise>" \
    --model_name_or_path "${METAROOT}" \
    --data_path "${DATAROOT}/unified_task_dailytalk_train_balanced.jsonl" \
    --val_data_path "${DATAROOT}/unified_task_dailytalk_test_balanced.jsonl" \
    --test_data_path "${DATAROOT}/unified_task_dailytalk_valid_balanced.jsonl" \
    --val_set_size 0 \
    --cache_dir ${CACHEROOT} \
    --preprocessing_num_workers 10 \
    --model_max_length 2048 \
    --bf16 True \
    --do_train \
    --do_eval \
    --do_predict True \
    --train_on_inputs True \
    --output_dir "${OUTROOT}" \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --eval_accumulation_steps 10 \
    --num_train_epochs 3 \
    --eval_strategy "steps" \
    --eval_steps 40 \
    --save_strategy "steps" \
    --save_steps 40 \
    --load_best_model_at_end True \
    --greater_is_better False \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.05 \
    --lr_scheduler_type "cosine" \
    --log_level debug \
    --overwrite_output_dir \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \

