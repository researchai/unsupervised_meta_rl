import os
import argparse

targets = [
    'RL2PPO_garage_ML1-push-v1',
    # 'RL2PPO_garage_ML1-push-v1_individual',
]

indices = [
    [11, 12, 13, 14, 15, 16, 17, 18],
]

parser = argparse.ArgumentParser()
parser.add_argument('--load', action='store_true', default=False)
parser.add_argument('--test', action='store_true', default=False)

args = parser.parse_args()
load_file = args.load
run_test = args.test

os.system('mkdir results')
# Download pickle files
for target, indice in zip(targets, indices):
    if isinstance(indice, int):
        for i, ind in enumerate(range(indice)):
            path = 'aws s3 cp --recursive s3://resl-garage-paper/{}_{} results/{}_{}'.format(
                target, ind, target, i)
            if load_file:
                os.system(path)
                print('Path: ', path)
    else:
        for ind in indice:
            path = 'aws s3 cp --recursive s3://resl-garage-paper/{}_{} results/{}_{}'.format(
                target, ind, target, ind)
            if load_file:
                os.system(path)
                print('Path: ', path)

print("Finish loading pickle files")

import pdb
pdb.set_trace()
cmd = 'python tests/benchmarks/garage/tf/algos/test_benchmark_rl2_meta_test_ml1.py '
cmd += '--test-rollouts 20 '
cmd += '--max-path-length 150 '
cmd += '--parallel 90 '
cmd += '--stride 10 '
for target, indice in zip(targets, indices):
    if isinstance(indice, int):
        for ind in range(indice):
            cmd += 'results/{}_{} '.format(target, ind)
    else:
        for ind in indice:
            cmd += 'results/{}_{} '.format(target, ind)
if run_test:
    os.system(cmd)
    print(cmd)