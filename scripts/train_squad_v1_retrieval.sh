#!/usr/bin/sh

MODEL_PATH=$1
OUTPUT_DIR=$2

echo $OUTPUT_DIR
mkdir -p $OUTPUT_DIR

printf '\n\n\n\nTASK: squad_v1_retrieval\n'
python engine/run_retrieval.py \
    --model_name_or_path $MODEL_PATH \
    --trust_remote_code true \
    --task_type "retrieval" \
    --pooler_type "first_token" \
    --dataset_name "rajpurkar/squad" \
    --dataset_config_name 'plain_text' \
    --text_column_names "question,context" \
    --do_train \
    --do_eval \
    --top_k "10" \
    --max_query_length "96" \
    --max_doc_length "256" \
    --per_device_train_batch_size "16" \
    --per_device_eval_batch_size "128" \
    --num_train_epochs 3 \
    --learning_rate "2e-5" \
    --evaluation_strategy "epoch" \
    --save_steps "-1" \
    --report_to none \
    --output_dir $OUTPUT_DIR/squad_v1_retrieval
