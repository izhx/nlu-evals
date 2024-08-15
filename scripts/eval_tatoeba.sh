#!/usr/bin/sh

MODEL_PATH=$1  # squad_v1_retrieval checkpoint
OUTPUT_DIR=$2
pooler_type=${3:-"first_token"}
is_r=${4:-0}

if [ "$is_r" -eq 1 ]; then
    max_query_length=96
    max_doc_length=256
    pooler_type=first_token
    name=tatoeba-xtreme-r
    if [ ! -f "$OUTPUT_DIR/squad_v1_retrieval/config.json" ]; then
        bash scripts/train_squad_v1_retrieval.sh $MODEL_PATH $OUTPUT_DIR
    fi
    MODEL_PATH=$OUTPUT_DIR/squad_v1_retrieval
else
    max_query_length=512
    max_doc_length=512
    name=tatoeba_$pooler_type
fi

echo $OUTPUT_DIR
mkdir -p $OUTPUT_DIR

printf '\nTASK: '$name'\n'
python engine/run_retrieval.py \
    --model_name_or_path $MODEL_PATH \
    --trust_remote_code true \
    --task_type "bitext" \
    --pooler_type $pooler_type \
    --dataset_name "mteb/tatoeba-bitext-mining" \
    --text_column_names "sentence1,sentence2" \
    --dataset_config_name "ara-eng" \
    --test_config_names "ara-eng,heb-eng,vie-eng,ind-eng,jav-eng,tgl-eng,eus-eng,mal-eng,tam-eng,tel-eng,afr-eng,nld-eng,deu-eng,ell-eng,ben-eng,hin-eng,mar-eng,urd-eng,tam-eng,fra-eng,ita-eng,por-eng,spa-eng,bul-eng,rus-eng,jpn-eng,kat-eng,kor-eng,tha-eng,swh-eng,cmn-eng,kaz-eng,tur-eng,est-eng,fin-eng,hun-eng,pes-eng,aze-eng,lit-eng,pol-eng,ukr-eng,ron-eng" \
    --do_predict \
    --top_k "10" \
    --max_query_length $max_query_length \
    --max_doc_length $max_doc_length \
    --per_device_eval_batch_size "128" \
    --report_to none \
    --output_dir $OUTPUT_DIR/$name
