import re
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
    parser.add_argument("-e", "--embedding_dim", default=64, type=int, help="embedding dim")
    parser.add_argument("-H", "--hidden_dim", default=32, type=int, help="hidden dim")
    parser.add_argument("-n", "--num_classes", default=2, type=int, help="num classes")
    parser.add_argument("-v", "--vocab_size", default=10002, type=int, help="vocab size")
    parser.add_argument("-m", "--max_epoch", default=1, type=int, help="num epoch")
    parser.add_argument("-t", "--name", default=None, type=str, help="name of the model")
    parser.add_argument("-l", "--num_layers", default=2, type=int, help="num layer for GRU")
    parser.add_argument("-u", "--num_labeled", default=200, type=int, help="num layer for GRU")
    args = parser.parse_args()

    # device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(88)

    with open('data_local/processed/data_{}.pickle'.format(args.num_labeled), 'rb') as handle:
        d = pickle.load(handle)

    model = create_model(d["args"])
    model = model.to(device)

    train_loader = d["train_loader"]
    criterion = nn.CrossEntropyLoss(reduction="none")
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    print(evaluate(model, d["val_loader"], device))
    train(d["groundtruth_loader"], d["val_loader"], model, optimizer, criterion, device, d["args"])


if __name__ == "__main__":
    main()
