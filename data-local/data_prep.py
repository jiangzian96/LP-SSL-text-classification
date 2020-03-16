import pandas as pd
import numpy as np
import pickle
from utils import *
import torch
from torch.utils.data import Dataset, DataLoader

df = pd.read_csv("data-local/spam.csv", usecols=["v1", "v2"], encoding='latin-1')

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

# make unlabeled
labeled_texts, train_labels, unlabeled_texts, unlabeled_labels = create_unlabeled(train_texts, train_labels, num_labeled=250)

# string ---> tokens
labeled_processed = tokenize_data(labeled_texts)
unlabeled_processed = tokenize_data(unlabeled_texts)
val_processed = tokenize_data(val_texts)

# create vocab
token2id, id2token = create_vocab(labeled_processed + unlabeled_processed, max_vocab=10000)

# tokens --- > indices
labeled_indices, train_labels = transform(labeled_processed, train_labels, token2id)
unlabeled_indices, unlabeled_labels = transform(unlabeled_processed, unlabeled_labels, token2id)
val_indices, val_labels = transform(val_processed, val_labels, token2id)

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
print("Creating dataloaders done!")

# save
torch.save(unlabeled_loader, 'data-local/processed/unlabeled_loader.pth')
torch.save(train_loader, 'data-local/processed/train_loader.pth')
torch.save(val_loader, 'data-local/processed/val_loader.pth')
with open('data-local/processed/token2id.pickle', 'wb') as handle:
  pickle.dump(token2id, handle, protocol=pickle.HIGHEST_PROTOCOL)
