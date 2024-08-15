#!/usr/bin/sh

MODEL_PATH=$1
OUTPUT_DIR=$2

echo $OUTPUT_DIR
mkdir -p $OUTPUT_DIR

if [ ! -f "$OUTPUT_DIR/squad_v1/config.json" ]; then
    bash scripts/run_squad.sh $MODEL_PATH $OUTPUT_DIR 1
fi

# TODO: xquad.hi 80
printf '\n\n\n\nTASK: xquad\n'
python engine/run_qa.py \
    --model_name_or_path $OUTPUT_DIR/squad_v1 \
    --trust_remote_code true \
    --dataset_name "google/xquad" \
    --dataset_config_name "xquad.en" \
    --test_config_names "xquad.ar,xquad.de,xquad.el,xquad.en,xquad.es,xquad.ru,xquad.th,xquad.tr,xquad.vi,xquad.zh,xquad.ro" \
    --test_split "validation" \
    --do_predict \
    --doc_stride "128" \
    --max_seq_length "384" \
    --per_device_eval_batch_size "128" \
    --report_to "none" \
    --output_dir $OUTPUT_DIR/xquad
