import argparse
import yaml
from easydict import EasyDict
import os
import time


def get_args():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument(
        '--algo', default='dqn_per', help='algorithm to use: dqn')

    parser.add_argument(
        '--config', default='./experiments/dqn_per/config.yaml',
        help='config')

    args = parser.parse_args()
    return args


if __name__ == '__main__':

    for i in list(range(5))[::-1]:
        print(i + 1)
        time.sleep(1)
    print('start.....')

    args = get_args()
    if args.algo == 'dqn':
        from agents.dqn.runner import Runner
    elif args.algo == 'dqn_per':
        from agents.dqn_per.runner import Runner

    with open(args.config) as f:
        config = EasyDict(yaml.load(f, yaml.FullLoader))
    config.log_root = os.path.dirname(args.config)

    runner = Runner(config)
    runner.run()
