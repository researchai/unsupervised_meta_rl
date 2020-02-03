import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import tests.helpers as Rh

def plot_garage_vs_promp_multiple_trial():
	g_x = 'TotalEnvSteps'
	g_y = 'Evaluation/AverageReturn'
	p_x = 'n_timesteps'
	p_y = 'train-AverageReturn'

	env_id = 'HalfCheetahDirEnv'
	a = 'garage'
	a_names = ['RL2PPO-garage-HalfCheetahDir_0', 'RL2PPO-garage-HalfCheetahDir_1', 'RL2PPO-garage-HalfCheetahDir_2']
	a_trials = ['trial_1_seed_57', 'trial_1_seed_81', 'trial_1_seed_74']
	b = 'promp'
	b_names = ['RL2PPO-ProMP-HalfCheetahDir_0', 'RL2PPO-ProMP-HalfCheetahDir_1', 'RL2PPO-ProMP-HalfCheetahDir_2']
	b_trials = ['trial_1_seed_81', 'trial_1_seed_10', 'trial_1_seed_58']

	garage_tf_csvs = [
		'/Users/wongtsankwong/Desktop/garage_results/{}/{}/{}/{}/progress.csv'.format(
		a_name, env_id, a_trial, a) for a_name, a_trial in zip(a_names, a_trials)
	]
	promp_csvs = [
		'/Users/wongtsankwong/Desktop/garage_results/{}/{}/{}/{}/progress.csv'.format(
		b_name, env_id, b_trial, b) for b_name, b_trial in zip(b_names, b_trials)
	]

	plt_file = 'result.png'
	Rh.relplot(g_csvs=garage_tf_csvs,
	           b_csvs=promp_csvs,
	           g_x=g_x,
	           g_y=g_y,
	           g_z=a,
	           b_x=p_x,
	           b_y=p_y,
	           b_z=b,
	           trials=3,
	           seeds=[0,0,0],
	           plt_file=plt_file,
	           env_id='HalfCheetahDir over 3 seeds')

def plot_garage_variation(b_dir, b_trial, b_name, meta_train=True):
	g_x = 'TotalEnvSteps'
	if meta_train:
		g_y = 'Evaluation/AverageReturn'
	else:
		g_y = 'MetaTest/AverageReturn'

	env_id = 'HalfCheetahDirEnv'
	a = 'garage'
	a_names = ['RL2PPO-garage-HalfCheetahDir_0', 'RL2PPO-garage-HalfCheetahDir_1', 'RL2PPO-garage-HalfCheetahDir_2']
	a_trials = ['trial_1_seed_57', 'trial_1_seed_81', 'trial_1_seed_74']

	garage_tf_csvs = [
		'/Users/wongtsankwong/Desktop/garage_results/{}/{}/{}/{}/progress.csv'.format(
		a_name, env_id, a_trial, a) for a_name, a_trial in zip(a_names, a_trials)
	]
	garage_tf_csvs2 = [
		'/Users/wongtsankwong/Desktop/garage_results/{}/{}/{}/{}/progress.csv'.format(
		b_dir, env_id, b_trial, a) for _ in range(len(garage_tf_csvs))
	]

	plt_file = 'result.png'
	Rh.relplot(g_csvs=garage_tf_csvs,
	           b_csvs=garage_tf_csvs2,
	           g_x=g_x,
	           g_y=g_y,
	           g_z=a,
	           b_x=g_x,
	           b_y=g_y,
	           b_z=b_name,
	           trials=3,
	           seeds=[0,0,0],
	           plt_file=plt_file,
	           env_id='HalfCheetahVel')

# plot_garage_vs_promp_multiple_trial()
plot_garage_variation(b_dir='RL2PPO-garage-HalfCheetahDir_individual', b_trial='trial_1_seed_38', b_name='garage-individual', meta_train=False)
# plot_garage_variation(b_dir='halfcheetahvel-individual', b_trial='trial_1_seed_68', b_name='garage-individual-path', meta_train=False)
# plot_garage_variation(b_dir='halfcheetahvel-individual', b_trial='trial_1_seed_68', b_name='garage-individual-path', meta_train=False)

