#!/bin/bash
# CUDA_VISIBLE_DEVICES=4 PYTHONPATH="../../" bash .sh
# ps | grep -ie python | awk '{print $1}' | xargs kill -9 

defaults="--identifier all -n 24 -f 0 --attack NA --LT --use-cuda"
for seed in 0 1 2
do
    for agg in "cm" "cp" "rfa" "krum" "avg"
    do
        python exp_gan.py $defaults --agg $agg --bucketing $s --seed $seed &
        pids[$!]=$!

        python exp_gan.py $defaults --agg $agg --noniid --bucketing $s --seed $seed &
        pids[$!]=$!
    done

    # wait for all pids
    for pid in ${pids[*]}; do
        wait $pid
    done
    unset pids
done