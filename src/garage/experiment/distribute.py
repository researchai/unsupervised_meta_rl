import subprocess

from fabric import Connection


def get_root_path():
    git_root_path = subprocess.check_output(
        ('git', 'rev-parse', '--show-toplevel'), stderr=subprocess.DEVNULL)
    return git_root_path.decode('utf-8').strip()


def make_launcher_archive(git_root_path):
    """Saves an archive of the launcher's git repo to the log directory.

    Args:
        git_root_path (str): Absolute path to git repo to archive.

    """
    files_to_archive = subprocess.check_output(
        ('git', 'ls-files', '--others', '--exclude-standard', '--cached',
         '-z'),
        cwd=git_root_path).strip()
    archive_path = 'launch_archive.tar.xz'
    subprocess.run(('tar', '--null', '--files-from', '-', '--auto-compress',
                    '--create', '--file', archive_path),
                   input=files_to_archive,
                   cwd=git_root_path,
                   check=True)


def distribute(config, exp_name_suffix, connect_kwargs):
    git_root_path = get_root_path()
    make_launcher_archive(git_root_path)
    remote_dir = f'garage_exp_{exp_name_suffix}'
    for host, exp in config.items():
        print(f'Connecting to {host}')
        c = Connection(host=host, connect_kwargs=connect_kwargs)
        _run(c, f'mkdir {remote_dir}')
        c.put(f'{git_root_path}/launch_archive.tar.xz', remote=remote_dir)
        with c.prefix(
            'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco200/bin'
        ):
            with c.cd(remote_dir):
                _run(c, 'tar -xf launch_archive.tar.xz')
                _run(c, 'python3 -m venv garage_venv')
                with c.prefix('source garage_venv/bin/activate'):
                    _run(c, 'pip install --upgrade pip')
                    _run(c, 'pip install .[all]')
                    _run(c, 'pip install .[dev]')
                    _run(c, f'tmux new -d -s {remote_dir}')
                    _run(c,
                         f'tmux send-keys -t {remote_dir}'
                         f' \'source garage_venv/bin/activate && ./{exp}\''
                         f' ENTER')


# TODO let caller pass echo and hide
def _run(connection, command, echo=True, hide=False, **kwargs):
    connection.run(command, echo=echo, hide=hide, **kwargs)
