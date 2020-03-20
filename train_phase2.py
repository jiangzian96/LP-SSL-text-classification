import argparse
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from train_utils import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total_epochs", default=99, type=int, help="total number of epochs to train with updating pseudo labels", required=True)
    parser.add_argument("--num_epochs", default=1, type=int, help="num epoch")
    parser.add_argument("--name", default="phase2", type=str, help="name of the phase2 model")
    parser.add_argument("--num_labeled", default=4250, type=int, help="number of labeled data used in make_data.py", required=True)
    parser.add_argument("--knn", default=100, type=int, help="k for knn")
    parser.add_argument("--prev_model_name", default="baseline", type=str, help="name of the baseline/phase1 model", required=True)
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(88)

    # load dataloaders
    with open("data_local/processed/data_{}.pickle".format(args.num_labeled), 'rb') as handle:
        d = pickle.load(handle)

    # path of the baseline/phase1 model
    PATH = "models/{}_model.pt".format(args.prev_model_name)

    # config of model
    model_config = torch.load(PATH, map_location=torch.device(device))["args"]

    # epoch 0
    batch_features = extract_features(d["groundtruth_loader"], model_path=PATH, device=device)
    p_labels, updated_weights, updated_class_weights = label_propagation(batch_features, d["groundtruth_labels"], d["labeled_idx"], d["unlabeled_idx"], k=args.knn)
    pseudo_loader = update_pseudoloader(d["all_indices"], p_labels, updated_weights, updated_class_weights)
    model = create_model(model_config)
    model.load_state_dict(torch.load(PATH, map_location=torch.device(device))["model_state_dict"])
    criterion = nn.CrossEntropyLoss(reduction="none")
    optimizer = optim.Adam(model.parameters())
    print("Epoch 0")
    train(pseudo_loader, d["val_loader"], model, optimizer, criterion, device, args)

    # path of the phase2 model
    PATH = "models/{}_model.pt".format(args.name)

    # epoch 1-T'
    for i in range(args.total_epochs):
        print("Epoch {}".format(i + 1))
        batch_features = extract_features(d["groundtruth_loader"], path=PATH, device=device)
        p_labels, updated_weights, updated_class_weights = label_propagation(batch_features, d["groundtruth_labels"], d["labeled_idx"], d["unlabeled_idx"], k=args.knn)
        pseudo_loader = update_pseudoloader(d["all_indices"], p_labels, updated_weights, updated_class_weights)
        model = create_model(model_config)
        model.load_state_dict(torch.load("models/{}_model.pt".format(args.name), map_location=torch.device(device))["model_state_dict"])
        criterion = nn.CrossEntropyLoss(reduction="none")
        optimizer = optim.Adam(model.parameters())
        train(pseudo_loader, d["val_loader"], model, optimizer, criterion, device, args)


if __name__ == "__main__":
    main()
