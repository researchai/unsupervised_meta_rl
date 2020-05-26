import subprocess

from fabric import Connection

# EXPORT_LD_LIBRARY_PATH = 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME' \
#                          '/.mujoco/mujoco200/bin '

GARAGE_GIT_URI = 'https://github.com/rlworkgroup/garage.git/'


def run_benchmark_remote(host, run_name_suffix, connect_kwargs,
                         *benchmark_names):
    remote_dir = f'garage_auto_benchmark{run_name_suffix}'
    print(f'Connecting to {host}')
    c = Connection(host=host, connect_kwargs=connect_kwargs)
    _run(c, f'mkdir {remote_dir}')
    # with c.prefix(EXPORT_LD_LIBRARY_PATH):
    with c.cd(remote_dir):
        _run(c, f'git clone -b auto_bench_remote {GARAGE_GIT_URI}')
        with c.cd('garage'):
            _run(c, f'tmux new -d -s {remote_dir}')
            benchmarks_to_run = ' '.join(benchmark_names)
            _run(c, (f'tmux send-keys -l -t {remote_dir} '
                     f'\'make run-benchmarks '
                     f'RUN_CMD="garage_benchmark run '
                     f'{benchmarks_to_run}"\' '))
            _run(c, f'tmux send-keys -t {remote_dir} ENTER')

            # _run(c, 'make run-benchmarks RUN_CMD=')
            # _run(c, f'python3 -m venv garage_venv')
            # _run(c, 'pip3 install --upgrade pip')
            # _run(c, f'pip3 install .[all]')
            # _run(c, f'pip3 install .[dev]')
            # with c.cd('benchmarks'):
            #     _run(c, 'pip install -e .')
            #     _run(c, f'tmux new -d -s {remote_dir}')
            #     _run(c, f'tmux send-keys -t {remote_dir} garage_benchmark '
            #             f'run {" ".join(benchmark_names)} ENTER')


# TODO let caller pass echo and hide
def _run(connection, command, echo=True, hide=False, **kwargs):
    connection.run(command, echo=echo, hide=hide, **kwargs)


if __name__ == '__main__':
    run_benchmark_remote('gitanshu@172.18.1.3', '_trial_13', None,
                         'auto_ppo_benchmarks', 'auto_trpo_benchmarks')
