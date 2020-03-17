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
    # device
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device('cpu')
    torch.manual_seed(88)

    labeled_loader, _, unlabeled_loader, _, _ = create_dataloaders(num_labeled=100)
    PATH = "models/baseline_model.pt"
    batch_features = extract_features(labeled_loader, unlabeled_loader, path=PATH, device=device)
    X, distances, indices = get_knn(batch_features, k=50)
    W = get_W(X, distances, indices, gamma=3)
    print(W[0])


if __name__ == "__main__":
    main()
