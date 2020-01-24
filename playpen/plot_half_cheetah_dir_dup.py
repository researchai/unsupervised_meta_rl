from tests import benchmark_helper

promp_fit_first = [
    '/home/kzhu/prog/resl/garage/tests/benchmarks/garage/torch/algos/data/local/benchmarks/maml/2020-01-05-23-22-16-837812-important/HalfCheetahVel/trial_1_seed_50/promp/progress.csv',
    '/home/kzhu/prog/resl/garage/tests/benchmarks/garage/torch/algos/data/local/benchmarks/maml/2020-01-06-14-10-18-248858/HalfCheetahVel/trial_1_seed_12/promp/progress.csv'
]

promp_shared_baselines = [
    '/home/kzhu/prog/resl/garage/tests/benchmarks/garage/torch/algos/data/local/benchmarks/maml/2020-01-07-11-12-44-047076-garage-predict-first/HalfCheetahVel/trial_1_seed_58/promp/progress.csv',
    '/home/kzhu/prog/resl/garage/tests/benchmarks/garage/torch/algos/data/local/benchmarks/maml/2020-01-07-11-12-44-047076-garage-predict-first/HalfCheetahVel/trial_2_seed_78/promp/progress.csv'
]
garage_independent_baselines = [
    '/home/kzhu/prog/resl/garage/tests/benchmarks/garage/torch/algos/data/local/benchmarks/maml/2020-01-08-15-20-36-012224-promp-predict-first/HalfCheetahVel/trial_1_seed_93/promp/progress.csv',
    '/home/kzhu/prog/resl/garage/tests/benchmarks/garage/torch/algos/data/local/benchmarks/maml/2020-01-08-17-05-40-650200-promp-predict-first/HalfCheetahVel/trial_1_seed_79/promp/progress.csv']

benchmark_helper.plot_average_over_trials(
    [promp_fit_first, promp_shared_baselines, garage_independent_baselines],
    ys=[
        'Step_1-AverageReturn',
        'Step_1-AverageReturn',
        'Step_1-AverageReturn'
    ],
    xs=['n_timesteps', 'n_timesteps', 'n_timesteps'],
    plt_file='/home/kzhu/prog/resl/garage/playpen/benchmark_dup.png',
    env_id='HalfCheetahDir',
    x_label='',
    y_label='AverageReturn',
    names=['ProMP_fit-first', 'ProMP_fit-later_shared-baseline', 'ProMP_fit-later_independent-baseline']
)
