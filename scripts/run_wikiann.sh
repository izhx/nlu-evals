#!/usr/bin/sh

MODEL_PATH=$1
OUTPUT_DIR=$2

echo $OUTPUT_DIR
mkdir -p $OUTPUT_DIR

printf "\n\nTASK: wikiann\n"
python engine/run_ner.py \
    --model_name_or_path $MODEL_PATH \
    --trust_remote_code true \
    --dataset_name "unimelb-nlp/wikiann" \
    --dataset_config_name "en" \
    --test_config_names "ar,he,vi,id,jv,ms,tl,eu,ml,ta,te,af,nl,en,de,el,bn,hi,mr,ur,fa,fr,it,pt,es,bg,ru,ja,ka,ko,th,sw,yo,my,zh,kk,tr,et,fi,hu,qu,pl,uk,az,lt,pa,gu,ro" \
    --evaluation_strategy "epoch" \
    --save_steps "-1" \
    --do_train \
    --do_eval \
    --do_predict \
    --max_seq_length "128" \
    --num_train_epochs "10" \
    --learning_rate "2e-5" \
    --per_device_train_batch_size "32" \
    --per_device_eval_batch_size "128" \
    --report_to "none" \
    --output_dir $OUTPUT_DIR/wikiann
