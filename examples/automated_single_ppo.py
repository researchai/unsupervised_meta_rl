#!/usr/bin/env python3

import argparse
import os
import subprocess

from garage.experiment import to_local_command

from multiworld... import env_list

if __name__ == '__main__':
    for env in env_list:
        pass
        command = ['python','..py']
        print(command)
        try:
            subprocess.call(command, shell=True, env=os.environ)
        except Exception as e:
            print(e)
            if isinstance(e, KeyboardInterrupt):
                raise
