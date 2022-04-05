# main.py -- Module to build the networks and start the training/testing
#
# Usage:
#   python3 main.py --realTime {0 | 1} --train {0 | 1}

import argparse
import sys

# Custom module imports
# TODO: Import data-loader
import net.net as net


def parse_args(args):
    """
    Builds the argument-parser and parses the arguments

    Returns:
        (realTime?, train?)
    """
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--realTime', type=int, required=True, choices=(0, 1))
    arg_parser.add_argument('--train', type=int, required=True, choices=(0, 1))

    # Parse the arguments and return the appropriate values
    got_args = arg_parser.parse_args(args)

    return got_args.realTime, got_args.train


if __name__ == '__main__':
    real_time_mode, train_mode = parse_args(sys.argv[1:])

    # TODO: Prepare the data-loader object

    # Backbone network for extracting the features
    # NOTE: The output number of channels from the backbone network must be 64
    # TODO: Add the real-time feature extractor network here
    network = net.InterpolationNet(real_time_mode)
