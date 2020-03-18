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
    # device
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device('cpu')
    torch.manual_seed(88)

    _, val_loader, _, _, pseudo_loader, p_labels, ground_truth_labels, all_indices = create_dataloaders(num_labeled=100)
    PATH = "models/baseline_model.pt"
    batch_features = extract_features(pseudo_loader, path=PATH, device=device)
    p_labels, weights, class_weights = run_LP(batch_features, p_labels)
    #pseudo_loader = update_loader(all_indices, p_labels, weights, class_weights)
    pseudo_dataset = SpamDataset(all_indices, p_labels, 128)
    pseudo_loader = torch.utils.data.DataLoader(dataset=pseudo_dataset,
                                                batch_size=32,
                                                collate_fn=pseudo_dataset.spam_collate_func,
                                                shuffle=False)
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
    model.load_state_dict(torch.load("models/baseline_model.pt")["model_state_dict"])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.1)
    train_without_weights(pseudo_loader, val_loader, model, optimizer, criterion, device, args)


if __name__ == "__main__":
    main()
