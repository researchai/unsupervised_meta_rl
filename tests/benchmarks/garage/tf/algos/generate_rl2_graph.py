import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import tests.helpers as Rh


class TestNothing:
	def test_nothing_here(self):
		# ######## generate plot for one garage vs one promp
		# g_x = 'TotalEnvSteps'
		# g_y = 'Evaluation/AverageReturn'
		# p_x = 'n_timesteps'
		# p_y = 'train-AverageReturn'

		# df_g = pd.read_csv('data/local/experiment/garage_half_cheetah_vel_epi=2_itr=500/progress.csv')
		# df_g['Type'] = 'Garage'

		# # df_p = pd.read_csv('data/local/experiment/promp_epi=2_itr=1000.csv')
		# df_p = pd.read_csv('data/local/experiment/progress.csv')
		# df_p['Type'] = 'ProMP'
		# df_p = df_p.rename(columns={p_x: g_x, p_y: g_y})
		# data = pd.concat([df_g, df_p])

		# ax = sns.relplot(x=g_x, y=g_y, hue='Type', kind='line', data=data)
		# ax.axes.flatten()[0].set_title('HalfCheetahRandVel')

		# plt.savefig('data/local/experiment/garage_vs_promp_epi=10_itr=500_HalfCheetahRandDir.png')

		# plt.close()

		######## generate plot for x garage vs x promp
		# target_folder = 'HalfCheetahRandVel_benchmark/'
		# seeds = [48, 75, 96]
		# prefix = 'data/local/benchmarks/rl2/'+target_folder
		# garage_tf_csvs = [
		# 	prefix+'trial_{}_seed_{}/garage/progress.csv'.format(i+1, seed) for i, seed in enumerate(seeds)
		# ]
		# promp_csvs = [
		# 	prefix+'trial_{}_seed_{}/promp/progress.csv'.format(i+1, seed) for i, seed in enumerate(seeds)
		# ]
		# garage_tf_csvs2 = prefix +'progress.csv'
		# plt_file = prefix+'benchmark_halfcheetahRandVel_result.png'
		# Rh.relplot(g_csvs=garage_tf_csvs,
		#            b_csvs=promp_csvs,
		#            g_csvs2='data/local/experiment/new_garage_epi=10_itr=500_baselines_sample_norm_adv/progress.csv',
		#            g_x='TotalEnvSteps',
		#            g_y='AverageReturn',
		#            g_z='Garage-previous',
		#            g_z2='Garage-latest',
		#            b_x='n_timesteps',
		#            b_y='train-AverageReturn',
		#            b_z='ProMP',
		#            trials=len(seeds),
		#            seeds=seeds,
		#            plt_file=plt_file,
		#            env_id='HalfCheetahRandDir over 3 seeds',
		#            x_label='TotalEnvSteps',
		#            y_label='AverageReturn')

		##### plot promp fit baseline before and after predict
		# p_x = 'n_timesteps'
		# p_y = 'train-AverageReturn'

		# df_g = pd.read_csv('data/local/experiment/promp_epi=2_itr=1000_fit_baseline_after.csv')
		# df_g['Type'] = 'ProMP-2'

		# df_p = pd.read_csv('data/local/experiment/promp_epi=2_itr=1000.csv')
		# df_p['Type'] = 'ProMP'
		# data = pd.concat([df_g, df_p])

		# ax = sns.relplot(x=p_x, y=p_y, hue='Type', kind='line', data=data)
		# ax.axes.flatten()[0].set_title('HalfCheetah')

		# plt.savefig('data/local/experiment/result_return.png')

		# plt.close()

		#### plot promp fit baseline before and after predict

		# g_x = 'TotalEnvSteps'
		# g_y = 'AverageReturn'

		# df_g = pd.read_csv('data/local/experiment/garage_epi=10_itr=500_new_baselines_new_adv_fit_baseline_first/progress.csv')
		# df_g['Type'] = 'Garage-fit-first'

		# df_p = pd.read_csv('data/local/experiment/garage_epi=10_itr=500_new_baselines_new_adv_fit_baseline_after/progress.csv')
		# df_p['Type'] = 'Garage-fit-after'
		# data = pd.concat([df_g, df_p])

		# ax = sns.relplot(x=g_x, y=g_y, hue='Type', kind='line', data=data)
		# ax.axes.flatten()[0].set_title('HalfCheetahRandVel')

		# plt.savefig('data/local/experiment/result_return.png')

		# plt.close()


		######## Only plot promp
		# g_x = 'n_timesteps'
		# g_y = 'train-AverageReturn'
		# # seeds = [2, 16, 28]
		# # df_g = [pd.read_csv('data/local/benchmarks/rl2/fit_baseline_after_epi=1_promp/HalfCheetah/trial_{}_seed_{}/promp/progress.csv'.format(i+1, seed)) for (i, seed) in enumerate(seeds)]
		# # df_g = pd.concat(df_g, axis=0)
		# df_g = pd.read_csv('/home/tsan/Desktop/ProMP/data/rl2/test_509/progress.csv')
		# df_g['Type'] = 'ProMP'
		# ax = sns.relplot(x=g_x, y=g_y, hue='Type', kind='line', data=df_g)
		# ax.axes.flatten()[0].set_title('HalfCheetah')

		# plt.savefig('data/local/experiment/result_return.png')

		# plt.close()

		# ###### Only plot garage
		g_x = 'TotalEnvSteps'
		# g_y = 'Evaluation/AverageReturn'
		g_y = 'SuccessRate'
		# df_g = pd.read_csv('data/local/experiment/ml10-rl2/progress.csv')
		df_g = pd.read_csv('../progress.csv')
		df_g['Type'] = 'Garage'
		ax = sns.relplot(x=g_x, y=g_y, hue='Type', kind='line', data=df_g)
		ax.axes.flatten()[0].set_title('pick-place')

		plt.savefig('data/local/experiment/result_return3.png')

		plt.close()