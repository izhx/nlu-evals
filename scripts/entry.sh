#!/usr/bin/sh

MODEL_PATH=$1
OUTPUT_DIR=$2
EVAL_GROUP=$3

echo $OUTPUT_DIR
mkdir -p $OUTPUT_DIR


# mnli rte mrpc stsb xnli ~= 5000
if [ $EVAL_GROUP -eq 0 ]; then
    for task in mnli ; do
        bash scripts/run_glue_one.sh $MODEL_PATH $OUTPUT_DIR $task
    done
    for task in rte mrpc stsb ; do
        bash scripts/run_glue_one.sh $MODEL_PATH $OUTPUT_DIR $task mnli
    done
    bash scripts/eval_xnli.sh $MODEL_PATH $OUTPUT_DIR

# qnli qqp ~= 5200
elif [ $EVAL_GROUP -eq 1 ]; then
    for task in qnli qqp ; do
        bash scripts/run_glue_one.sh $MODEL_PATH $OUTPUT_DIR $task
    done

# wnli cola sst2 600 +  xquad 120  mlqa 375 (squad_v1 3850)  tydiqa 200  bucc 50  tatoeba 150 ~= 5000
elif [ $EVAL_GROUP -eq 2 ]; then
    for task in wnli cola sst2 ; do
        bash scripts/run_glue_one.sh $MODEL_PATH $OUTPUT_DIR $task
    done
    for run_job in eval_bucc18 eval_tatoeba run_tydiqa eval_mlqa eval_xquad ; do
        bash scripts/$run_job.sh $MODEL_PATH $OUTPUT_DIR
    done

# paws-x wikiann udpos ~= 2150
elif [ $EVAL_GROUP -eq 3 ]; then
    for run_job in run_pawsx run_wikiann run_udpos ; do
        bash scripts/$run_job.sh $MODEL_PATH $OUTPUT_DIR
    done
    bash scripts/run_udpos.sh $MODEL_PATH $OUTPUT_DIR "v2.7"

# mewslix 3900 + xcopa 10      siqa  1255      copa 10  ~= 5200
elif [ $EVAL_GROUP -eq 4 ]; then
    for run_job in eval_xcopa run_mewslix ; do
        bash scripts/$run_job.sh $MODEL_PATH $OUTPUT_DIR
    done

# lareqa (+squad_v1r) tatoeba ~= 4900
elif [ $EVAL_GROUP -eq 5 ]; then
    bash scripts/eval_lareqa.sh $MODEL_PATH $OUTPUT_DIR
    bash scripts/eval_tatoeba.sh $MODEL_PATH $OUTPUT_DIR _ 1

fi
