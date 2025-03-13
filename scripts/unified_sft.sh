#!/bin/bash

# METAROOT="/home/jhwan98/EmoSDS/SpeechGPT/speechgpt/llama/3_2/3B/Llama-3.2-3B-Instruct"
# METAROOT="/shared/NAS_SSD/jhl/futureinternet/output/asr_6layer_k1000_merged"
# METAROOT="/shared/NAS_SSD/jhl/futureinternet/output/stage3/checkpoint-3600"
# METAROOT="/shared/NAS_SSD/jhl/futureinternet/output/asr_ser_6layer/checkpoint-340"
# METAROOT="/shared/NAS_SSD/jhl/futureinternet/output/asr_ser_6layer_k1000_merged/checkpoint-445"
# METAROOT="/shared/NAS_SSD/jhl/futureinternet/output/asr_ser_7layer_k2000_merged2/checkpoint-742"
# METAROOT="/shared/NAS_SSD/jhl/futureinternet/output/asr_ser_7layer_k2000_diverse_prompt_no_dailytalk_10layer/checkpoint-424"
# METAROOT="/shared/NAS_SSD/jhl/futureinternet/output/asr_ser_7layer_k2000_diverse_prompt_only_esd/checkpoint-330"
# METAROOT="/shared/NAS_SSD/jhl/futureinternet/output/asr_ser_6layer_k1000_diverse_prompt_only_esd/checkpoint-385"
# METAROOT="/shared/NAS_SSD/jhl/futureinternet/output/asr_ser_6layer_k1000_diverse_prompt_only_esd_residual"
# METAROOT="/shared/NAS_SSD/jhl/futureinternet/output/asr_ser_6layer_k1000_diverse_prompt_dailytalk_finetune_newresidual"
# METAROOT="/shared/NAS_SSD/jhl/futureinternet/output/asr_ser_6layer_k1000_diverse_prompt_only_dailytalk_residual"
# METAROOT="/shared/data_zfs/jhwan/futureinternet/output/asr_ser_6layer_k1000_diverse_prompt_only_esd_residual_useLlama_20250220"
# METAROOT="/shared/NAS_SSD/jhl/futureinternet/output/unified_6layer_k1000_merged_dailytalk_balanced_5emotions_20250215/checkpoint-46"
# METAROOT="/shared/NAS_SSD/jhl/futureinternet/output/unified_6layer_k1000_merged_dailytalk_balanced_5emotions_newresidual_20250215/checkpoint-16"
# METAROOT="/shared/NAS_SSD/jhl/futureinternet/output/unified_6layer_k1000_merged_esd_balanced_type1_useStage1model_20250216/checkpoint-693"
# METAROOT="/shared/NAS_SSD/jhl/futureinternet/output/unified_6layer_k1000_merged_esd_balanced_type1_useLlama_20250219/checkpoint-693"
# METAROOT="/shared/NAS_SSD/jhl/futureinternet/output/unified_6layer_k1000_merged_esd_balanced_type1_useLlama_20250219/checkpoint-231"
METAROOT="/shared/NAS_SSD/jhl/futureinternet/output/unified_6layer_k1000_merged_only_esd_residual_only_type1_20250207/checkpoint-924"
# DATAROOT="data/unified/layer6_k1000_downsampled"
DATAROOT="data/unified/layer6_k1000_merged"
# DATAROOT="data/unified/layer7_k2000_merged2"
OUTROOT="/shared/data_zfs/jhwan/futureinternet/output/unified_6layer_k1000_merged_only_esd_residual_only_type1_20250207_checkpoint-924"
# OUTROOT="/shared/data_zfs/jhwan/futureinternet/output/unified_6layer_k1000_merged_esd_balanced_type1_onlyStage2-3_20250220"
CACHEROOT="${DATAROOT}/cache/"


mkdir -p ${CACHEROOT}/tokenized/train/
mkdir -p ${CACHEROOT}/tokenized/valid/

echo "================================= unified fine-tuning ================================="


# --nproc_per_node를 사용할 gpu 개수로 설정하면 됨
export CUDA_VISIBLE_DEVICES=1,2
torchrun \
    --nproc_per_node 2 \
    --standalone \
src/train/sft_residual.py \
    --train_task "unified" \
    --style_token_list "<anger> <happiness> <neutral> <sadness> <surprise>" \
    --model_name_or_path "${METAROOT}" \
    --data_path "${DATAROOT}/unified_task_esd_5emotions_test_balanced_type1.jsonl" \
    --val_data_path "${DATAROOT}/unified_task_esd_5emotions_valid_balanced_type1.jsonl" \
    --test_data_path "${DATAROOT}/unified_task_esd_5emotions_test_balanced_type1.jsonl" \
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
    --num_train_epochs 0 \
    --eval_strategy "steps" \
    --eval_steps 16 \
    --save_strategy "steps" \
    --save_steps 16 \
    --load_best_model_at_end True \
    --metric_for_best_model "res_style_UA" \
    --greater_is_better True \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.05 \
    --lr_scheduler_type "cosine" \
    --log_level debug \
    --logging_steps 1 \
    --overwrite_output_dir \
    --fsdp "full_shard" \
    --fsdp_config "/home/jhwan98/EmoSDS/scripts/fsdp_config.json" \
    --ddp_find_unused_parameters False

