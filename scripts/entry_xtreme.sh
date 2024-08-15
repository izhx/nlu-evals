#!/usr/bin/sh

MODEL_PATH=$1
OUTPUT_DIR=$2
EVAL_GROUP=${3:-"0"}

echo $OUTPUT_DIR
mkdir -p $OUTPUT_DIR


echo "EVAL_GROUP is "$EVAL_GROUP
# 默认全跑
if [ $EVAL_GROUP -eq 0 ]; then
    for run_job in eval_bucc18 eval_tatoeba run_tydiqa eval_mlqa eval_xquad run_pawsx run_wikiann run_udpos eval_xnli ; do
        bash scripts/$run_job.sh $MODEL_PATH $OUTPUT_DIR
    done

# paws-x wikiann udpos ~= 2150
elif [ $EVAL_GROUP -eq 1 ]; then
    for run_job in run_pawsx run_wikiann run_udpos ; do
        bash scripts/$run_job.sh $MODEL_PATH $OUTPUT_DIR
    done

else
    bash scripts/$EVAL_GROUP.sh $MODEL_PATH $OUTPUT_DIR
fi

