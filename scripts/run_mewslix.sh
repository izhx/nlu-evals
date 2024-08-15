#!/usr/bin/sh

MODEL_PATH=$1
OUTPUT_DIR=$2
MODE=${3:-"train_eval"}

echo $OUTPUT_DIR
mkdir -p $OUTPUT_DIR

printf '\n\n\n\nTASK: mewsli-x\n'

train() {
    python engine/run_retrieval.py \
        --model_name_or_path $MODEL_PATH \
        --trust_remote_code true \
        --task_type "retrieval" \
        --pooler_type "first_token" \
        --dataset_name "izhx/mewsli-x" \
        --dataset_config_name "wikipedia_pairs" \
        --text_column_names "mention,entity" \
        --do_train \
        --do_eval \
        --top_k "20" \
        --max_seq_length "64" \
        --per_device_train_batch_size "64" \
        --per_device_eval_batch_size "128" \
        --num_train_epochs 2 \
        --learning_rate "2e-5" \
        --evaluation_strategy "epoch" \
        --save_steps "-1" \
        --report_to none \
        --output_dir $OUTPUT_DIR/mewslix
}

# Auto multi-gpu
evaluate() {
    accelerate launch engine/run_retrieval.py \
        --model_name_or_path $OUTPUT_DIR/mewslix \
        --trust_remote_code true \
        --task_type "retrieval" \
        --pooler_type "first_token" \
        --dataset_name "izhx/mewsli-x" \
        --dataset_config_name "candidate_entities" \
        --text_column_names "mention,entity" \
        --corpus_text_column_name "description" \
        --corpus_split "test" \
        --test_config_names "ar,de,en,es,fa,ja,pl,ro,ta,tr,uk" \
        --positive_id_column_name "entity_id" \
        --merge_test true \
        --do_predict \
        --top_k "20" \
        --max_seq_length "64" \
        --per_device_train_batch_size "64" \
        --per_device_eval_batch_size "128" \
        --report_to none \
        --output_dir $OUTPUT_DIR/mewslix
}


echo "Mode is "$MODE

if [[ "$MODE" == *"train"* ]]; then
    printf '\n train \n'
    train
fi

if [[ "$MODE" == *"eval"* ]]; then
    printf '\n eval \n'
    evaluate
fi
