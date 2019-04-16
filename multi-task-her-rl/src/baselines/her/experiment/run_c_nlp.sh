#!/bin/sh
#SBATCH --mincpus 20
#SBATCH -p longq
#SBATCH -t 24:00:00
#SBATCH -e run_c_nlp.sh.err
#SBATCH -o run_c_nlp.sh.out
rm log.txt; 
export EXP_INTERP='/cm/shared/apps/intel/composer_xe/python3.5/intelpython3/bin/python3' ;
$EXP_INTERP train.py --env MultiTaskFetchArmNLP-v0 --num_cpu 20 --trial_id 0  &
wait