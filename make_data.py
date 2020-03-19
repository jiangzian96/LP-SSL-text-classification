import argparse
import os
import shutil
import time
import math
import pickle
import pdb
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.optim as optim

from data_local.utils import *
from train_utils import *


def main():
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument("-u", "--num_labeled", default=200, type=int, help="num layer for GRU")
    args = parser.parse_args()

    d = create_dataloaders(num_labeled=args.num_labeled)
    with open('data_local/processed/data_{}.pickle'.format(args.num_labeled), 'wb') as handle:
        pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
