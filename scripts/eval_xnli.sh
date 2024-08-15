#!/usr/bin/sh

MODEL_PATH=$1  # mnli checkpoint
OUTPUT_DIR=$2

echo $OUTPUT_DIR
mkdir -p $OUTPUT_DIR

if [ ! -f "$OUTPUT_DIR/mnli/config.json" ]; then
    bash scripts/run_glue_one.sh $MODEL_PATH $OUTPUT_DIR mnli
fi

printf '\n\n\n\nTASK: xnli\n'
python engine/run_xnli.py \
    --model_name_or_path $OUTPUT_DIR/mnli \
    --trust_remote_code true \
    --language "en" \
    --do_eval \
    --do_predict \
    --max_seq_length "128" \
    --per_device_eval_batch_size "128" \
    --report_to "none" \
    --output_dir $OUTPUT_DIR/xnli
