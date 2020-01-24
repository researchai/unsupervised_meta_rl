from tests import benchmark_helper

promp_csvs = [
    '/home/kzhu/prog/resl/garage/tests/benchmarks/garage/torch/algos/data/local/benchmarks/maml/2020-01-07-11-12-44-047076-garage-predict-first/HalfCheetahVel/trial_1_seed_58/promp/progress.csv',
    '/home/kzhu/prog/resl/garage/tests/benchmarks/garage/torch/algos/data/local/benchmarks/maml/2020-01-07-11-12-44-047076-garage-predict-first/HalfCheetahVel/trial_2_seed_78/promp/progress.csv',
    '/home/kzhu/prog/resl/garage/tests/benchmarks/garage/torch/algos/data/local/benchmarks/maml/2020-01-07-11-12-44-047076-garage-predict-first/HalfCheetahVel/trial_3_seed_54/promp/progress.csv'
]
garage_csvs = [
    '/home/kzhu/prog/resl/garage/tests/benchmarks/garage/torch/algos/data/local/benchmarks/maml/2020-01-08-15-20-36-012224-promp-predict-first/HalfCheetahVel/trial_1_seed_93/promp/progress.csv',
    '/home/kzhu/prog/resl/garage/tests/benchmarks/garage/torch/algos/data/local/benchmarks/maml/2020-01-08-17-05-40-650200-promp-predict-first/HalfCheetahVel/trial_1_seed_79/promp/progress.csv',
    '/home/kzhu/prog/resl/garage/tests/benchmarks/garage/torch/algos/data/local/benchmarks/maml/2020-01-08-17-05-40-650200-promp-predict-first/HalfCheetahVel/trial_2_seed_22/promp/progress.csv']

benchmark_helper.plot_average_over_trials(
    [promp_csvs, promp_csvs, garage_csvs, garage_csvs],
    ys=[
        'Step_0-AverageReturn', 'Step_1-AverageReturn',
        'Update_0/AverageReturn', 'Update_1/AverageReturn'
    ],
    xs=['n_timesteps', 'n_timesteps', 'TotalEnvSteps', 'TotalEnvSteps'],
    plt_file='/home/kzhu/prog/resl/garage/playpen/benchmark.png',
    env_id='HalfCheetahDir',
    x_label='',
    y_label='AverageReturn',
    names=['ProMP_pre-adapted', 'ProMP_post-adapted', 'garage_pre-adapted', 'garage_post-adapted']
)
