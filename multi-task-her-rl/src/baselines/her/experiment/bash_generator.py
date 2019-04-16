#!/usr/bin/python
# -*- coding: utf-8 -*-

import datetime

PATH_TO_INTERPRETER = "/cm/shared/apps/intel/composer_xe/python3.5/intelpython3/bin/python3"  # plafrim
# PATH_TO_INTERPRETER = "python"  # plafrim

env = 'MultiTaskFetchArmNLP-v0'
trial_id = list(range(0, 1))

script_path = './'
filename = 'run_c_nlp.sh'
filepath = script_path + filename
with open(filepath, 'w') as f:
    f.write('#!/bin/sh\n')
    f.write('#SBATCH --mincpus 20\n')
    f.write('#SBATCH -p longq\n')
    f.write('#SBATCH -t 24:00:00\n')
    f.write('#SBATCH -e ' + filename + '.err\n')
    f.write('#SBATCH -o ' + filename + '.out\n')
    f.write('rm log.txt; \n')
    f.write("export EXP_INTERP='%s' ;\n" % PATH_TO_INTERPRETER)

    for seed in trial_id:
        name = ("trial %s, %s" % (str(seed), str(datetime.datetime.now()))).title()
        f.write("$EXP_INTERP train.py --env %s --num_cpu 20 --trial_id %s  & \n" % (env, str(seed)))
    f.write('wait')