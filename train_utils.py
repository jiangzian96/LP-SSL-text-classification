import re
import argparse
import os
import shutil
import time
import math

import pdb

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
import scipy


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_classes, bidirectional=True, dropout_prob=0.1, num_layers=1):
        super().__init__()
        self.embedding_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=0)
        self.gru = nn.GRU(input_size=embedding_dim, hidden_size=hidden_size, bidirectional=bidirectional, dropout=dropout_prob, batch_first=True, num_layers=num_layers)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, inputs):
        out = self.embedding_layer(inputs)
        gru_out, h_n = self.gru(out)
        #(batch_size, seq_length, hidden_size*2)
        hidden_size_ = gru_out.size(2) // 2

        # maxpooling
        forward = gru_out[:, :, :hidden_size_]
        backward = gru_out[:, :, hidden_size_:]
        out = torch.stack([forward, backward])
        out, _ = torch.max(out, dim=2)
        out, _ = torch.max(out, dim=0)
        logits = self.fc(out)
        return logits


def create_model(args):
    vocab_size = args.vocab_size
    embedding_dim = args.embedding_dim
    hidden_dim = args.hidden_dim
    num_classes = args.num_classes

    return GRUClassifier(vocab_size=vocab_size, embedding_dim=embedding_dim, hidden_size=hidden_dim, num_classes=num_classes)


def evaluate(model, dataloader, device):
    accuracy = None
    model.eval()
    correct = 0
    total = 0
    print("Evaluating model...")
    for data_batch, labels_batch in tqdm(dataloader):
        data_batch = data_batch.to(device)
        labels_batch = labels_batch.to(device)
        outputs = F.softmax(model(data_batch), dim=1)
        predicted = outputs.max(1, keepdim=True)[1]
        total += labels_batch.size(0)
        correct += predicted.eq(labels_batch.view_as(predicted)).sum().item()
    accuracy = correct / total
    print("Evaluation done! accuracy: {}".format(accuracy))
    return accuracy


def train_without_weights(train_loader, val_loader, model, optimizer, criterion, device, args):
    train_loss_history = []
    val_accuracy_history = []
    best_val_acc = 0
    max_epoch = args.max_epoch
    name = args.name
    model.train()
    for epoch in tqdm(range(max_epoch)):
        for i, (data_batch, batch_labels) in enumerate(train_loader):
            preds = model(data_batch.to(device))
            loss = criterion(preds, batch_labels.to(device))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_loss_history.append(loss.item())
            if i % 50 == 0:
                print("loss: ", loss.item())
        accuracy = evaluate(model, val_loader, device)
        val_accuracy_history.append(accuracy)
        if accuracy > best_val_acc:
            best_val_acc = accuracy
            torch.save({
                "model_state_dict": model.state_dict(),
                "train_loss_history": train_loss_history,
                "val_accuracy_history": val_accuracy_history,
                "args": args
            }, "models/{}_model.pt".format(name))


def extract_features(labeled_loader, unlabeled_loader, path, device):
    args = torch.load(path)["args"]
    feature_extractor = create_model(args)
    feature_extractor.load_state_dict(torch.load(path)["model_state_dict"])
    feature_extractor.fc = Identity()
    feature_extractor.eval()
    res = torch.tensor([])
    print("Extracting features......")
    for i, (data_batch, batch_labels) in enumerate(tqdm(labeled_loader)):
        batch_features = feature_extractor(data_batch.to(device))
        res = torch.cat((res, batch_features), 0)

    for i, (data_batch, batch_labels) in enumerate(tqdm(unlabeled_loader)):
        batch_features = feature_extractor(data_batch.to(device))
        res = torch.cat((res, batch_features), 0)

    print("Extracted {} points each {}-dimensional!".format(*res.shape))
    return res


def get_knn(batch_features, k=5):
    X = batch_features.cpu().detach().numpy()
    print("Calculating k nearest neighbors for each point...")
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='brute').fit(X)
    distances, indices = nbrs.kneighbors(X)
    indices = indices[:, 1:]
    distances = distances[:, 1:]
    return X, distances, indices


def get_W(X, distances, indices, gamma=3, k=50):
    N = X.shape[0]
    A = np.zeros((N, N))
    row_idx = np.arange(N)
    row_idx_rep = np.tile(row_idx, (k, 1)).T
    print(row_idx_rep.flatten("F").shape)
    values = distances ** gamma
    W = scipy.sparse.csr_matrix((distances.flatten('F'), (row_idx_rep.flatten('F'), indices.flatten('F'))), shape=(N, N))
    W = W + W.T
    W = W - scipy.sparse.diags(W.diagonal())
    S = W.sum(axis=1)
    S[S == 0] = 1
    D = np.array(1. / np.sqrt(S))
    D = scipy.sparse.diags(D.reshape(-1))
    W_normalized = D * W * D
    return W_normalized
