#!/usr/bin/env python3
import os
import subprocess
from metaworld.envs.mujoco.env_dict import ALL_ENVIRONMENTS

names = list(ALL_ENVIRONMENTS.keys())
names = names[:25]
for i, name in enumerate(names):
    gpu_id = i % 4
    conda_args = ['source ~/miniconda3/etc/profile.d/conda.sh; conda activate garage; python sac_metaworldv2_test.py --gpu {} --env {} && python -v'.format(gpu_id, name)]
    FNULL = open(os.devnull, 'w')
    p = subprocess.Popen(conda_args, shell=True, executable="/bin/bash", stdout=FNULL, stderr=subprocess.STDOUT)
