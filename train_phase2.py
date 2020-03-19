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
from sklearn.semi_supervised import LabelPropagation

from data_local.utils import *
from train_utils import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--embedding_dim", default=64, type=int, help="embedding dim")
    parser.add_argument("-H", "--hidden_dim", default=32, type=int, help="hidden dim")
    parser.add_argument("-n", "--num_classes", default=2, type=int, help="num classes")
    parser.add_argument("-f", "--feature_extractor", default=False, type=bool, help="fc or Identity for last layer")
    parser.add_argument("-v", "--vocab_size", default=10002, type=int, help="vocab size")
    parser.add_argument("-m", "--max_epoch", default=1, type=int, help="num epoch")
    parser.add_argument("-t", "--name", default=None, type=str, help="name of the model")
    parser.add_argument("-l", "--num_layers", default=2, type=int, help="num layer for GRU")
    parser.add_argument("-u", "--num_labeled", default=200, type=int, help="num layer for GRU")
    parser.add_argument("-k", "--knn", default=500, type=int, help="k for kNN")
    args = parser.parse_args()

    # device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(88)

    with open("data_local/processed/data_{}.pickle".format(args.num_labeled), 'rb') as handle:
        d = pickle.load(handle)
    PATH = "models/baseline_model.pt"

    batch_features = extract_features(d["groundtruth_loader"], path=PATH, device=device)
    p_labels, updated_weights, updated_class_weights = run_LP(batch_features, d["groundtruth_labels"], d["labeled_idx"], d["unlabeled_idx"], k=args.knn)
    pseudo_loader = update_pseudoloader(d["all_indices"], p_labels, updated_weights, updated_class_weights)

    model = create_model(args)
    model.load_state_dict(torch.load("models/baseline_model.pt")["model_state_dict"])
    criterion = nn.CrossEntropyLoss(reduction="none")
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    train(pseudo_loader, val_loader, model, optimizer, criterion, device, args)


if __name__ == "__main__":
    main()
