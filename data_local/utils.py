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


def create_dataloaders(num_labeled=200):
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
    labeled_texts, train_labels, unlabeled_texts, unlabeled_labels = create_unlabeled(train_texts, train_labels, num_labeled=num_labeled)

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
                                               shuffle=False)

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
