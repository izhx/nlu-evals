#!/usr/bin/sh

MODEL_PATH=$1
OUTPUT_DIR=$2

echo $OUTPUT_DIR
mkdir -p $OUTPUT_DIR

printf "\n\nTASK: tydiqa\n"
python engine/run_qa.py \
    --model_name_or_path $MODEL_PATH \
    --trust_remote_code true \
    --dataset_name "juletxara/tydiqa_xtreme" \
    --dataset_config_name "en" \
    --test_config_names "en,ar,bn,fi,id,ko,ru,sw,te" \
    --doc_stride "128" \
    --save_steps "-1" \
    --do_train \
    --do_predict \
    --max_seq_length "384" \
    --num_train_epochs "3" \
    --learning_rate "3e-5" \
    --weight_decay "0.0001" \
    --per_device_train_batch_size "32" \
    --per_device_eval_batch_size "128" \
    --report_to "none" \
    --output_dir $OUTPUT_DIR/tydiqa
