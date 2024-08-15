#!/bin/bash

pattern=${1:-'results/lm/en-large-*/*'}

for path in $pattern ; do
    if [ -d "$path" ]; then
        echo $path
        rm -f $path/*.log
        for task in bucc18_first_token cola copa lareqa-xquadr mewslix mlqa \
mnli mrpc paws-x qnli qqp rte siqa squad_v1 squad_v1_retrieval sst2 stsb \
tatoeba_first_token tatoeba-xtreme-r tydiqa udpos udpos_v27 wikiann wnli xcopa \
xnli xquad ; do
            tp="$path/$task"
            if [ -d "$tp" ]; then
                mv $tp/all_results.json $tp.json
                rm -rf $tp
            fi
        done
    fi
done
