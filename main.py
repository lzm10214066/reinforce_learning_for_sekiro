import argparse
import yaml
from easydict import EasyDict
import os
from agents.dqn.runner import Runner


def get_args():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument(
        '--algo', default='dqn', help='algorithm to use: ddpg')

    parser.add_argument(
        '--config', default='./experiments/dqn/',
        help='config')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    with open(args.config) as f:
        config = EasyDict(yaml.load(f))
    config.log_root = os.path.dirname(args.config)

    runner = Runner(config)
    runner.run()
