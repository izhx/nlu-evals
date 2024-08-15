#!/usr/bin/sh

MODEL_PATH=$1
OUTPUT_DIR=$2
VERSION=${3:-"v2.5"}

printf "\n\n$OUTPUT_DIR\n"
mkdir -p $OUTPUT_DIR

if [ "$VERSION" == 'v2.5' ]; then
    printf "TASK: udpos $VERSION\n"
    dataset_name="google/xtreme"
    dataset_config_name="udpos.English"
    test_config_names="udpos.Afrikaans,udpos.Arabic,udpos.Basque,udpos.Bulgarian,udpos.Dutch,udpos.English,"\
"udpos.Estonian,udpos.Finnish,udpos.French,udpos.German,udpos.Greek,udpos.Hebrew,udpos.Hindi,udpos.Hungarian,"\
"udpos.Indonesian,udpos.Italian,udpos.Japanese,udpos.Kazakh,udpos.Korean,udpos.Chinese,udpos.Marathi,udpos.Persian,"\
"udpos.Portuguese,udpos.Russian,udpos.Spanish,udpos.Tagalog,udpos.Tamil,udpos.Telugu,udpos.Thai,udpos.Turkish,"\
"udpos.Urdu,udpos.Vietnamese,udpos.Yoruba"
    OUTPUT_DIR=$OUTPUT_DIR/udpos

elif [ "$VERSION" == 'v2.7' ]; then
    printf "TASK: udpos $VERSION\n"
    dataset_name="izhx/xtreme-r-udpos"
    dataset_config_name="en"
    test_config_names="af,ar,bg,de,el,en,es,et,eu,fa,fi,fr,he,hi,hu,id,it,ja,kk,ko,lt,mr,nl,pl,pt,ro,ru,ta,te,th,tl,tr,uk,ur,vi,wo,yo,zh"
    OUTPUT_DIR=$OUTPUT_DIR/udpos_v27
fi

python engine/run_ner.py \
    --model_name_or_path $MODEL_PATH \
    --trust_remote_code true \
    --task_name "pos" \
    --dataset_name $dataset_name \
    --dataset_config_name $dataset_config_name \
    --test_config_names $test_config_names \
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
    --output_dir $OUTPUT_DIR
