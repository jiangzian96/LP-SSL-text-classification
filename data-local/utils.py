from collections import Counter
import pandas as pd
import numpy as np
import pickle as pkl
import sacremoses
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset, dataloader


def create_unlabeled(train_texts, train_labels, num_labeled=100):
    print("Building unlabeled data......")
    spam_texts = train_texts[train_labels == 1]
    spam_labels = train_labels[train_labels == 1]
    ham_texts = train_texts[train_labels == 0]
    ham_labels = train_labels[train_labels == 0]
    spam_labels[num_labeled:] = -1
    ham_labels[num_labeled:] = -1
    spam_labels, ham_labels = list(spam_labels), list(ham_labels)
    spam_texts, ham_texts = list(spam_texts), list(ham_texts)
    labeled_texts = spam_texts[:num_labeled] + ham_texts[:num_labeled]
    train_labels = spam_labels[:num_labeled] + ham_labels[:num_labeled]
    unlabeled_texts = spam_texts[num_labeled:] + ham_texts[num_labeled:]
    unlabeled_labels = [-1 for _ in range(len(unlabeled_texts))]

    assert (len(labeled_texts) == 2 * num_labeled == len(train_labels))
    assert (sum(np.array(train_labels) == 1) == num_labeled)
    assert (sum(np.array(train_labels) == 0) == num_labeled)
    print("Building unlabeled data done!")
    return labeled_texts, train_labels, unlabeled_texts, unlabeled_labels


def tokenize_data(data):
    # input: list of strings
    # return: list of list of tokens

    tokenizer = sacremoses.MosesTokenizer()
    preprocessed_data = []
    print("Preprocessing data into tokens......")
    for sent in tqdm(data):
        tokenized_sent = tokenizer.tokenize(sent.lower())
        preprocessed_data.append(tokenized_sent)

    return preprocessed_data


def create_vocab(preprocessed_data, max_vocab=10000):
    # input: list of list of tokens
    # output: token2id: dict, id2token: list

    all_tokens = []
    PAD_IDX = 0
    UNK_IDX = 1
    print("Building vocab......")
    for tokens in tqdm(preprocessed_data):
        for token in tokens:
            all_tokens.append(token)
    token_counter = Counter(all_tokens)
    vocab, count = zip(*token_counter.most_common(max_vocab))
    token2id = dict(zip(vocab, range(2, 2 + len(vocab))))
    token2id["<PAD>"] = PAD_IDX
    token2id["<UNK>"] = UNK_IDX
    id2token = ["<PAD>", "<UNK>"] + list(vocab)
    return token2id, id2token


def transform(preprocessed_data, labels, token2id):
    # transform list of list of tokenes --> list of list of ids according to token2id
    print("Transforming tokens into indices......")
    text_indices = []
    for tokens in tqdm(preprocessed_data):
        indices = [token2id.get(token, 1) for token in tokens]
        text_indices.append(indices)
    return text_indices, labels


class SpamDataset(Dataset):
    def __init__(self, data_list, target_list, max_sent_length=128):
        self.data_list = data_list
        self.target_list = target_list
        self.max_sent_length = max_sent_length
        assert (len(self.data_list) == len(self.target_list))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, key, max_sent_length=None):
        if max_sent_length is None:
            max_sent_length = self.max_sent_length
        token_idx = self.data_list[key][:max_sent_length]
        label = self.target_list[key]
        return [token_idx, label]

    def spam_collate_func(self, batch):
        data_list = []
        label_list = []
        max_batch_seq_len = None
        length_list = []
        for datum in batch:
            label_list.append(datum[1])
            length_list.append(len(datum[0]))

        if max(length_list) < self.max_sent_length:
            max_batch_seq_len = max(length_list)
        else:
            max_batch_seq_len = self.max_sent_length

        for datum in batch:
            padded_vec = np.pad(np.array(datum[0]),
                                pad_width=((0, max_batch_seq_len - len(datum[0]))),
                                mode="constant", constant_values=0)
            data_list.append(padded_vec)

        return [torch.from_numpy(np.array(data_list)), torch.LongTensor(label_list)]
