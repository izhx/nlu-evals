#!/usr/bin/sh

MODEL_PATH=$1
OUTPUT_DIR=$2

echo $OUTPUT_DIR
mkdir -p $OUTPUT_DIR

printf "\n\nTASK: paws-x\n"
python engine/run_classification.py \
    --model_name_or_path $MODEL_PATH \
    --trust_remote_code true \
    --dataset_name "google-research-datasets/paws-x" \
    --dataset_config_name "en" \
    --test_config_names "de,en,es,fr,ja,ko,zh" \
    --paired_texts "true" \
    --text_column_names "sentence1,sentence2" \
    --shuffle_train_dataset \
    --metric_name "accuracy" \
    --evaluation_strategy "epoch" \
    --save_steps "-1" \
    --do_train \
    --do_eval \
    --do_predict \
    --max_seq_length "128" \
    --num_train_epochs "2" \
    --learning_rate "2e-5" \
    --per_device_train_batch_size "32" \
    --per_device_eval_batch_size "128" \
    --report_to "none" \
    --output_dir $OUTPUT_DIR/paws-x
