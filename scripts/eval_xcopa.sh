#!/usr/bin/sh

MODEL_PATH=$1
OUTPUT_DIR=$2

echo $OUTPUT_DIR
mkdir -p $OUTPUT_DIR

if [ ! -f "$OUTPUT_DIR/copa/config.json" ]; then
    if [ ! -f "$OUTPUT_DIR/siqa/config.json" ]; then
        bash scripts/train_siqa.sh $MODEL_PATH $OUTPUT_DIR
    fi
    bash scripts/run_copa.sh $OUTPUT_DIR/siqa $OUTPUT_DIR
fi

printf '\n\n\n\nTASK: xcopa\n'
python engine/run_multiple_choice.py \
    --model_name_or_path $OUTPUT_DIR/copa \
    --trust_remote_code true \
    --dataset_name "cambridgeltl/xcopa" \
    --dataset_config_name "zh" \
    --test_config_names "et,ht,id,it,qu,sw,ta,th,tr,vi,zh" \
    --text_column_names "premise,question" \
    --ending_names "choice1,choice2" \
    --question_in_first "true" \
    --first_input_template "{context} What was the {question}?" \
    --do_eval \
    --do_predict \
    --max_seq_length "128" \
    --per_device_eval_batch_size "128" \
    --report_to "none" \
    --output_dir $OUTPUT_DIR/xcopa
