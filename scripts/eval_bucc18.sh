#!/usr/bin/sh

MODEL_PATH=$1
OUTPUT_DIR=$2
pooler_type=${3:-"first_token"}

echo $OUTPUT_DIR
mkdir -p $OUTPUT_DIR

printf '\n\nTASK: bucc18\n'
python engine/run_retrieval.py \
    --model_name_or_path $MODEL_PATH \
    --trust_remote_code true \
    --task_type "bitext" \
    --pooler_type $pooler_type \
    --dataset_name "google/xtreme" \
    --text_column_names "source_sentence,target_sentence" \
    --dataset_config_name "bucc18.zh" \
    --test_config_names "bucc18.de,bucc18.fr,bucc18.ru,bucc18.zh" \
    --do_eval \
    --do_predict \
    --top_k "10" \
    --max_seq_length "512" \
    --per_device_eval_batch_size "128" \
    --report_to none \
    --output_dir $OUTPUT_DIR/bucc18_$pooler_type
