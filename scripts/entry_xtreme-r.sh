#!/usr/bin/sh

MODEL_PATH=$1
OUTPUT_DIR=$2
EVAL_GROUP=${3:-"0"}

echo $OUTPUT_DIR
mkdir -p $OUTPUT_DIR


echo "EVAL_GROUP is "$EVAL_GROUP
# 默认全跑
if [ $EVAL_GROUP -eq 0 ]; then
    for run_job in run_tydiqa run_wikiann run_udpos eval_xcopa run_mewslix eval_mlqa eval_xquad eval_xnli eval_lareqa ; do
        bash scripts/$run_job.sh $MODEL_PATH $OUTPUT_DIR
    done
    bash scripts/tatoeba.sh $MODEL_PATH $OUTPUT_DIR _ 1

# tydiqa lareqa (+squad_v1r) tatoeba ~= 4900  + xnli
elif [ $EVAL_GROUP -eq 1 ]; then
    for run_job in run_tydiqa eval_lareqa ; do
        bash scripts/$run_job.sh $MODEL_PATH $OUTPUT_DIR
    done
    bash scripts/tatoeba.sh $MODEL_PATH $OUTPUT_DIR _ 1
    bash scripts/eval_xnli.sh $MODEL_PATH $OUTPUT_DIR

# mlqa xquad 4400
elif [ $EVAL_GROUP -eq 2 ]; then
    for run_job in eval_mlqa eval_xquad ; do
        bash scripts/$run_job.sh $MODEL_PATH $OUTPUT_DIR
    done

# mewslix 3900
elif [ $EVAL_GROUP -eq 3 ]; then
    bash scripts/run_mewslix.sh $MODEL_PATH $OUTPUT_DIR

# wikiann udpos xcopa 2900
elif [ $EVAL_GROUP -eq 4 ]; then
    for run_job in run_wikiann eval_xcopa ; do
        bash scripts/$run_job.sh $MODEL_PATH $OUTPUT_DIR
    done
    bash scripts/run_udpos.sh $MODEL_PATH $OUTPUT_DIR "v2.7"

else
    bash scripts/$EVAL_GROUP.sh $MODEL_PATH $OUTPUT_DIR
fi

