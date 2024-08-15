#!/usr/bin/sh

MODEL_PATH=$1
OUTPUT_DIR=$2

echo $OUTPUT_DIR
mkdir -p $OUTPUT_DIR

if [ ! -f "$OUTPUT_DIR/squad_v1/config.json" ]; then
    bash scripts/run_squad.sh $MODEL_PATH $OUTPUT_DIR 1
fi

printf '\n\n\n\nTASK: mlqa\n'
python engine/run_qa.py \
    --model_name_or_path $OUTPUT_DIR/squad_v1 \
    --trust_remote_code true \
    --dataset_name "facebook/mlqa" \
    --dataset_config_name "mlqa.zh.zh" \
    --test_config_names "mlqa.ar.ar,mlqa.de.de,mlqa.en.en,mlqa.es.es,mlqa.hi.hi,mlqa.vi.vi,mlqa.zh.zh" \
    --do_predict \
    --doc_stride "128" \
    --max_seq_length "384" \
    --per_device_eval_batch_size "128" \
    --report_to "none" \
    --output_dir $OUTPUT_DIR/mlqa
