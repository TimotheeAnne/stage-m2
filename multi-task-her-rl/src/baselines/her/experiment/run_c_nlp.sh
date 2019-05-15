#!/bin/sh
#SBATCH --mincpus 20
#SBATCH -p longq
#SBATCH -t 24:00:00
#SBATCH -e run_c_nlp.sh.err
#SBATCH -o run_c_nlp.sh.out
rm log.txt; 
export EXP_INTERP='/home/tanne/anaconda3/envs/multiTask/bin/python3.6' ;
$EXP_INTERP train.py --num_cpu 20   &
wait
