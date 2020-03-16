import pandas as pd
import numpy as np
import pickle
import argparse
from utils import *
import torch
from torch.utils.data import Dataset, DataLoader


def create_dataloaders():
    df = pd.read_csv("data_local/spam.csv", usecols=["v1", "v2"], encoding='latin-1')

    # 1 - spam, 0 - ham
    df.v1 = (df.v1 == "spam").astype("int")

    # split into train and val
    # 0.85 vs 0.15

    val_size = int(df.shape[0] * 0.15)

    # Shuffle
    df = df.sample(frac=1)

    val_df = df[:val_size]
    train_df = df[val_size:]

    train_texts, train_labels = np.array(train_df.v2), np.array(train_df.v1, dtype="int")
    val_texts, val_labels = np.array(val_df.v2), np.array(val_df.v1, dtype="int")
    all_labels = np.copy(train_labels)

    # make unlabeled
    #parser = argparse.ArgumentParser()
    #parser.add_argument("-n", "--num_labeled", default=200, type=int, help="number of labeled example per class")
    #args = parser.parse_args()
    #NUM_LABELED = args.num_labeled
    labeled_texts, train_labels, unlabeled_texts, unlabeled_labels = create_unlabeled(train_texts, train_labels, num_labeled=NUM_LABELED)

    # string ---> tokens
    labeled_processed = tokenize_data(labeled_texts)
    unlabeled_processed = tokenize_data(unlabeled_texts)
    val_processed = tokenize_data(val_texts)
    all_processed = labeled_processed + unlabeled_processed

    # create vocab
    token2id, id2token = create_vocab(all_processed, max_vocab=10000)

    # tokens --- > indices
    labeled_indices, train_labels = transform(labeled_processed, train_labels, token2id)
    unlabeled_indices, unlabeled_labels = transform(unlabeled_processed, unlabeled_labels, token2id)
    val_indices, val_labels = transform(val_processed, val_labels, token2id)
    all_indices, all_labels = transform(all_processed, all_labels, token2id)

    # create dataloaders
    BATCH_SIZE = 32
    max_sent_length = 128
    print("Creating dataloaders......")
    train_dataset = SpamDataset(labeled_indices, train_labels, max_sent_length)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=BATCH_SIZE,
                                               collate_fn=train_dataset.spam_collate_func,
                                               shuffle=True)

    val_dataset = SpamDataset(val_indices, val_labels, train_dataset.max_sent_length)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                             batch_size=BATCH_SIZE,
                                             collate_fn=train_dataset.spam_collate_func,
                                             shuffle=False)

    unlabeled_dataset = SpamDataset(unlabeled_indices, unlabeled_labels, train_dataset.max_sent_length)
    unlabeled_loader = torch.utils.data.DataLoader(dataset=unlabeled_dataset,
                                                   batch_size=BATCH_SIZE,
                                                   collate_fn=unlabeled_dataset.spam_collate_func,
                                                   shuffle=False)

    all_dataset = SpamDataset(all_indices, all_labels, train_dataset.max_sent_length)
    all_loader = torch.utils.data.DataLoader(dataset=all_dataset,
                                             batch_size=BATCH_SIZE,
                                             collate_fn=all_dataset.spam_collate_func,
                                             shuffle=False)
    print("Creating dataloaders done!")

    return train_loader, val_loader, unlabeled_loader, all_loader, token2id
