#!/usr/bin/sh

MODEL_PATH=$1
OUTPUT_DIR=$2
TASK=$3

if [ -n "$4" ]; then
    MODEL_PATH=$OUTPUT_DIR/$4
    ignore_mismatched_sizes=true
else
    ignore_mismatched_sizes=false
fi

echo $OUTPUT_DIR
mkdir -p $OUTPUT_DIR

printf '\n\n\n\n TASK_NAME: '$TASK'\n'
python engine/run_glue.py \
    --model_name_or_path $MODEL_PATH \
    --trust_remote_code true \
    --ignore_mismatched_sizes $ignore_mismatched_sizes \
    --task_name $TASK \
    --do_train \
    --do_eval \
    --max_seq_length 128 \
    --per_device_train_batch_size 32 \
    --learning_rate 2e-5 \
    --warmup_ratio 0.06 \
    --num_train_epochs 3 \
    --save_steps -1 \
    --report_to none \
    --output_dir $OUTPUT_DIR/$TASK
