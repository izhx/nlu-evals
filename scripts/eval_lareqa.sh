#!/usr/bin/sh

MODEL_PATH=$1  # squad_v1_retrieval checkpoint
OUTPUT_DIR=$2

echo $OUTPUT_DIR
mkdir -p $OUTPUT_DIR

if [ ! -f "$OUTPUT_DIR/squad_v1_retrieval/config.json" ]; then
    bash scripts/train_squad_v1_retrieval.sh $MODEL_PATH $OUTPUT_DIR
fi

printf '\n\n\n\nTASK: lareqa-xquadr\n'
python engine/run_retrieval.py \
    --model_name_or_path $OUTPUT_DIR/squad_v1_retrieval \
    --trust_remote_code true \
    --task_type "retrieval" \
    --pooler_type "first_token" \
    --dataset_name "google-research-datasets/xquad_r" \
    --dataset_config_name "en" \
    --test_config_names "ar,de,el,en,es,hi,ru,th,tr,vi,zh" \
    --text_column_names "question,context" \
    --merge_test true \
    --test_split "validation" \
    --do_predict \
    --top_k "20" \
    --max_query_length "96" \
    --max_doc_length "256" \
    --per_device_eval_batch_size "128" \
    --report_to none \
    --output_dir $OUTPUT_DIR/lareqa-xquadr
