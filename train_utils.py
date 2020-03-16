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


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_classes, bidirectional=True, dropout_prob=0.1, feature_extractor=False, num_layers=1):
        super().__init__()
        self.embedding_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=0)
        self.gru = nn.GRU(input_size=embedding_dim, hidden_size=hidden_size, bidirectional=bidirectional, dropout=dropout_prob, batch_first=True, num_layers=num_layers)
        if not feature_extractor:
            self.fc = nn.Linear(hidden_size, num_classes)
        else:
            self.fc = Identity()

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
    feature_extractor = args.feature_extractor

    return GRUClassifier(vocab_size=vocab_size, embedding_dim=embedding_dim, hidden_size=hidden_dim, num_classes=num_classes, feature_extractor=feature_extractor)


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
                "val_accuracy_history": val_accuracy_history
            }, "models/{}_model.pt".format(name))
