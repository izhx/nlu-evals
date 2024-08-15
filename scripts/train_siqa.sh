#!/usr/bin/sh

MODEL_PATH=$1
OUTPUT_DIR=$2

echo $OUTPUT_DIR
mkdir -p $OUTPUT_DIR

printf '\n\n\n\nTASK: siqa\n'
python engine/run_multiple_choice.py \
    --model_name_or_path $MODEL_PATH \
    --trust_remote_code true \
    --dataset_name "allenai/social_i_qa" \
    --text_column_names "context,question" \
    --ending_names "answerA,answerB,answerC" \
    --question_in_first true \
    --do_train \
    --do_eval \
    --max_seq_length "128" \
    --num_train_epochs "5" \
    --learning_rate "2e-5" \
    --per_device_train_batch_size "32" \
    --per_device_eval_batch_size "128" \
    --evaluation_strategy "epoch" \
    --save_steps "-1" \
    --report_to none \
    --output_dir $OUTPUT_DIR/siqa
