import os

targets = [
	'RL2PPO_garage_ML45',
	'RL2PPO_garage_ML45_normalized-reward'
	'RL2PPO_garage_ML10'
	'RL2PPO_garage_ML1-reach-v1',
	'RL2PPO_garage_ML1-reach-v1_individual',
	'RL2PPO_garage_ML10_normalized-reward',
	'RL2PPO_garage_ML10_sample-9',
	'RL2PPO_garage_ML10_sample-8',
	'RL2PPO_garage_ML10_sample-6',
	'RL2PPO_garage_ML10_sample-2',
	'RL2PPO_garage_ML10_individual',
	'RL2PPO_garage_ML10_max-ent',
	'RL2TRPO_garage_ML10'
]

# total = 53
indices = [
	10,
	5,
	[11, 12, 13],
	3,
	5,
	3,
	3,
	3,
	3,
	5,
	5,
	5
]


os.system("mkdir results")
# Download pickle files
for target, indice in zip(targets, indices):
	if isinstance(indice, int):
		for i, ind in enumerate(range(indice)):
			os.system("aws s3 cp --recursive s3://resl-garge-paper/{}_{} results/{}_{}".format(target, ind, target, i))
	else:
		for i, ind in enumerate(indice):
			os.system("aws s3 cp --recursive s3://resl-garge-paper/{}_{} results/{}_{}".format(target, ind, target, i))
cmd = "python tests/benchmarks/garage/tf/algos/test_benchmark_rl2_meta_test_ml10.py "
cmd += "--test-rollouts 10 "
cmd += "--max-path-length 150 "
cmd += "--parallel 90 "
cmd += "--stride 10 "
for target, indice in zip(targets, indices):
	if isinstance(indice, int):
		for ind in range(indice):
			cmd += "{}_{} ".format(target, ind)
	else:
		for ind in indice:
			cmd += "{}_{} ".format(target, ind)
print(cmd)
