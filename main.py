# main.py -- Module to build the networks and start the training/testing
#
# Usage:
#   python3 main.py --realTime {0 | 1} --train {0 | 1}

import argparse
import pathlib
import sys
import torch.utils.data

# Custom module imports
import dataset.x4k as dataset
import net.net as net
import start


# --------------------------------- CONSTANTS ---------------------------------
CHKPT_DIR_PATH = './checkpoints'  # The directory for storing model checkpoints
OUTPUT_DIR_PATH = './output'      # The directory for storing output
DATA_DIR_PATH = './dataset'       # Root directory of the dataset
TRAIN_DIR = 'train'               # Training set directory name
TEST_DIR = 'test'                 # Testing set directory name

BATCH_SIZE = 16                   # No. of samples per batch
N_EPOCHS = 20                     # No. of epochs to train
CHKPT_EPOCHS = 2                  # Epochs after which model will be saved
# -----------------------------------------------------------------------------


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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    assert torch.cuda.is_available(), "[ERROR]: Need CUDA supported GPU or else it will not work :("

    # Set the dataset path accordingly
    chkpt_dir_path = pathlib.Path(CHKPT_DIR_PATH)
    output_dir_path = pathlib.Path(OUTPUT_DIR_PATH)
    data_dir_path = pathlib.Path(DATA_DIR_PATH)
    dataset_dir = TRAIN_DIR if train_mode else TEST_DIR
    dataset_path = data_dir_path / dataset_dir

    # Prepare the torch dataset object, and then the loader class
    dataset_obj = dataset.X4K1000FPS(dataset_path)
    data_loader = torch.utils.data.DataLoader(
        dataset_obj,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False
    )

    # Build the network
    network = net.InterpolationNet(real_time_mode)

    # Finally start the training/testing
    if train_mode:
        start.train(network, data_loader, N_EPOCHS, CHKPT_EPOCHS, chkpt_dir_path, device)
    else:
        # Testing mode. TODO: Yet to implement it
        start.test(network, data_loader, output_dir_path, device)

