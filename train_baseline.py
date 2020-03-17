import re
import argparse
import os
import shutil
import time
import math
import pickle
import pdb

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
    parser.add_argument("-e", "--embedding_dim", default=64, type=int, help="embedding dim")
    parser.add_argument("-H", "--hidden_dim", default=32, type=int, help="hidden dim")
    parser.add_argument("-n", "--num_classes", default=2, type=int, help="num classes")
    parser.add_argument("-f", "--feature_extractor", default=False, type=bool, help="fc or Identity for last layer")
    parser.add_argument("-v", "--vocab_size", default=10002, type=int, help="vocab size")
    parser.add_argument("-m", "--max_epoch", default=1, type=int, help="num epoch")
    parser.add_argument("-t", "--name", default=None, type=str, help="name of the model")
    args = parser.parse_args()

    # device
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device('cpu')
    torch.manual_seed(88)

    model = create_model(args)

    train_loader, val_loader, unlabeled_loader, all_loader, token2id = create_dataloaders(num_labeled=100)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.1)
    print(evaluate(model, val_loader, device))
    train_without_weights(train_loader, val_loader, model, optimizer, criterion, device, args)


if __name__ == "__main__":
    main()
