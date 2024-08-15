#!/usr/bin/sh

MODEL_PATH=$1
OUTPUT_DIR=$2
VERSION=${3:-"1"}

mkdir -p $OUTPUT_DIR

squad_v2() {
    printf '\n\n\n\n TASK_NAME: squad_v2\n'
    python engine/run_qa.py \
        --model_name_or_path $MODEL_PATH \
        --trust_remote_code true \
        --dataset_name rajpurkar/squad_v2 \
        --doc_stride 128 --version_2_with_negative \
        --do_train \
        --do_eval \
        --max_seq_length 384 \
        --per_device_train_batch_size 48 \
        --learning_rate 5e-5 \
        --num_train_epochs 2 --save_steps -1 \
        --report_to none \
        --output_dir $OUTPUT_DIR/squad_v2
}

squad_v1() {
    printf '\n\n\n\n TASK_NAME: squad_v1\n'
    python engine/run_qa.py \
        --model_name_or_path $MODEL_PATH \
        --trust_remote_code true \
        --dataset_name "rajpurkar/squad" \
        --dataset_config_name 'plain_text' \
        --doc_stride "128" \
        --evaluation_strategy "epoch" \
        --save_steps "-1" \
        --do_train \
        --do_eval \
        --max_seq_length "384" \
        --num_train_epochs "3" \
        --learning_rate "3e-5" \
        --weight_decay "0.0001" \
        --per_device_train_batch_size "32" \
        --per_device_eval_batch_size "128" \
        --report_to "none" \
        --output_dir $OUTPUT_DIR/squad_v1
}

if [ $VERSION -eq 1 ]; then
    squad_v1
else
    squad_v2
fi
