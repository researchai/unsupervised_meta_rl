import os

targets = [
    #'RL2PPO_garage_ML10',
    #'RL2PPO_garage_ML10_individual',
    #'RL2PPO_garage_ML10_max-ent',
    #'RL2TRPO_garage_ML10',
    #'RL2PPO_garage_ML1-reach-v1',
    #'RL2PPO_garage_ML1-reach-v1_individual',
    #'RL2PPO_garage_ML10_normalized-reward',
    'RL2PPO_garage_ML45_normalized-reward'
]

indices = [
    #5,
    #5,
    #5,
    #5,
    #[11, 12, 13],
    #3,
    #5,
    5,
]

load_file = True
run_test = True

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
        for i, ind in enumerate(indice):
            path = 'aws s3 cp --recursive s3://resl-garage-paper/{}_{} results/{}_{}'.format(
                target, ind, target, i)
            if load_file:
                os.system(path)
            print('Path: ', path)
import pdb
pdb.set_trace()
cmd = 'python tests/benchmarks/garage/tf/algos/test_benchmark_rl2_meta_test_ml45_normalized.py '
cmd += '--test-rollouts 10 '
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
