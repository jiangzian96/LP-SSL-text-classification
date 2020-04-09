import argparse
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from train_utils import *


def main():
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding_dim", default=300, type=int, help="embedding dim")
    parser.add_argument("--hidden_dim", default=32, type=int, help="hidden dim", required=True)
    parser.add_argument("--num_epochs", default=20, type=int, help="number of epochs", required=True)
    parser.add_argument("--name", default="baseline", type=str, help="name of the model", required=True)
    parser.add_argument("--num_layers", default=2, type=int, help="number of layers for GRU")
    parser.add_argument("--num_labeled", default=4250, type=int, help="number of labeled data used in make_data.py", required=True)
    parser.add_argument("--model_type", type=str, help="model type", required=True, choices=["gru", "bert"])
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(88)

    with open('data_local/processed/data_{}_{}.pickle'.format(args.model_type, args.num_labeled), 'rb') as handle:
        d = pickle.load(handle)

    if args.model_type == "gru":
        fname = "data_local/wiki-news-300d-1M.vec"
        vectors = load_vectors(fname, MAX_NUM=50000)
        weights_matrix = build_word_embeddings(d["id2token"], vectors)
        model = create_model(args, phase2=False, weights_matrix=weights_matrix)
        model = model.to(device)
        criterion = nn.CrossEntropyLoss(reduction="none")
        optimizer = optim.Adam(model.parameters())
        train(d["train_loader"], d["val_loader"], model, optimizer, criterion, device, args)
    elif args.model_type == "bert":
        model = create_model(args)
        model = model.to(device)
        criterion = nn.CrossEntropyLoss(reduction="none")
        optimizer = optim.Adam(model.parameters(), lr=2e-5, eps=1e-08)
        train(d["train_loader"], d["val_loader"], model, optimizer, criterion, device, args)


if __name__ == "__main__":
    main()
