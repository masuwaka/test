#!/bin/zsh

for fc in Branin2D ; do
    for sd in `seq 11 100` ; do
        for acq in Sobol CEI ; do
            echo $fc $sd $acq
            python bench.py -sd $sd -fc $fc -nstd 0.0 -l log -dev cpu -acq $acq -nwarm 10 -ns 100 -bs 1
        done
    done
done