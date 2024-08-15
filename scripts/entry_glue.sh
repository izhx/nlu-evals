#!/usr/bin/sh

MODEL_PATH=$1
OUTPUT_DIR=$2

if [ -n "$3" ]; then
    EVAL_GROUP=$3
else
    EVAL_GROUP=0
fi

echo $OUTPUT_DIR
mkdir -p $OUTPUT_DIR


echo "EVAL_GROUP is "$EVAL_GROUP
# 默认全跑
if [ $EVAL_GROUP -eq 0 ]; then
    for task in wnli cola sst2 qnli qqp mnli ; do
        bash scripts/run_glue_one.sh $MODEL_PATH $OUTPUT_DIR $task
    done
    for task in rte mrpc stsb ; do
        bash scripts/run_glue_one.sh $MODEL_PATH $OUTPUT_DIR $task mnli
    done

# sst2 mnli rte mrpc stsb ~= 5500
elif [ $EVAL_GROUP -eq 1 ]; then
    for task in mnli sst2 ; do
        bash scripts/run_glue_one.sh $MODEL_PATH $OUTPUT_DIR $task
    done
    for task in rte mrpc stsb ; do
        bash scripts/run_glue_one.sh $MODEL_PATH $OUTPUT_DIR $task mnli
    done

# wnli cola qnli qqp ~= 5300
elif [ $EVAL_GROUP -eq 2 ]; then
    for task in wnli cola qnli qqp ; do
        bash scripts/run_glue_one.sh $MODEL_PATH $OUTPUT_DIR $task
    done

fi
