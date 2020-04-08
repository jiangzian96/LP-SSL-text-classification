from collections import Counter
import pandas as pd
import numpy as np
import sacremoses
from tqdm import tqdm
import torch
import io
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer


def build_unlabeled(train_texts, train_labels, num_labeled=4250):
    print("Sampling unlabeled data......")

    n = len(train_texts)
    permute = np.random.permutation(n)
    unlabeled_idx = permute[num_labeled:]
    labeled_idx = permute[:num_labeled]
    all_texts = train_texts
    groundtruth_labels = np.copy(train_labels)
    unlabeled_texts = [train_texts[i] for i in unlabeled_idx]
    unlabeled_labels = [-1 for _ in range(len(unlabeled_texts))]
    labeled_texts = [train_texts[i] for i in labeled_idx]
    labeled_labels = [train_labels[i] for i in labeled_idx]

    return labeled_texts, labeled_labels, unlabeled_texts, unlabeled_labels, all_texts, groundtruth_labels, unlabeled_idx, labeled_idx


def tokenize_data(data, token_type):
    # input: list of strings
    # return: list of list of tokens
    if token_type == "gru":
        tokenizer = sacremoses.MosesTokenizer()
        preprocessed_data = []
        print("Processing data into tokens......")
        for sent in tqdm(data):
            tokenized_sent = tokenizer.tokenize(sent.lower())
            preprocessed_data.append(tokenized_sent)

    elif token_type == "bert":
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        preprocessed_data = []
        print("Processing data into tokens......")
        for sent in tqdm(data):
            tokenized_sent = ["[CLS]"] + tokenizer.tokenize(sent)
            # BERT only accepts sequences of max length 512
            preprocessed_data.append(tokenized_sent[:512])

    return preprocessed_data, tokenizer


def build_vocab(preprocessed_data, max_vocab=10000):
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


def transform(preprocessed_data, labels, token_type, tokenizer=None, token2id=None):
    # input: list of list of tokens
    # output: list of list of ids according to token2id

    print("Transforming tokens into indices......")
    if token_type == "gru":
        text_indices = []
        for tokens in tqdm(preprocessed_data):
            indices = [token2id.get(token, 1) for token in tokens]
            text_indices.append(indices)
    elif token_type == "bert":
        text_indices = list(map(tokenizer.convert_tokens_to_ids, preprocessed_data))
    return text_indices, labels


class ReviewDataset(Dataset):
    def __init__(self, data_list, target_list, weights_list, class_weights, max_sent_length):
        self.data_list = data_list
        self.target_list = target_list
        self.max_sent_length = max_sent_length
        self.weights_list = weights_list
        self.class_weights = class_weights
        assert (len(self.data_list) == len(self.target_list))
        assert (len(self.weights_list) == len(self.data_list))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, key, max_sent_length=None):
        if max_sent_length is None:
            max_sent_length = self.max_sent_length

        token_idx = self.data_list[key][:max_sent_length]
        label = self.target_list[key]
        w = self.weights_list[key]
        c = self.class_weights[label]

        return [token_idx, label, w, c]

    def spam_collate_func(self, batch):
        data_list = []
        label_list = []
        max_batch_seq_len = None
        length_list = []
        w_list = []
        c_list = []
        for datum in batch:
            label_list.append(datum[1])
            length_list.append(len(datum[0]))
            w_list.append(datum[2])
            c_list.append(datum[3])

        if max(length_list) < self.max_sent_length:
            max_batch_seq_len = max(length_list)
        else:
            max_batch_seq_len = self.max_sent_length

        for datum in batch:
            padded_vec = np.pad(np.array(datum[0]),
                                pad_width=((0, max_batch_seq_len - len(datum[0]))),
                                mode="constant", constant_values=0)
            data_list.append(padded_vec)

        return [torch.from_numpy(np.array(data_list)), torch.LongTensor(label_list), torch.FloatTensor(w_list), torch.FloatTensor(c_list)]


def build_dataloaders(args):
    num_labeled = args.num_labeled
    token_type = args.model_type

    # 50k reviews
    df = pd.read_csv("data_local/reviews.csv", usecols=["review", "sentiment"], encoding='latin-1')

    # 1 - pos, 0 - neg
    df.sentiment = (df.sentiment == "positive").astype("int")

    # 0.85 vs 0.15 train/val split
    val_size = int(df.shape[0] * 0.15)

    # Shuffle
    df = df.sample(frac=1)

    val_df = df[:val_size]
    train_df = df[val_size:]

    train_texts, train_labels = np.array(train_df.review), np.array(train_df.sentiment, dtype="int")
    val_texts, val_labels = np.array(val_df.review), np.array(val_df.sentiment, dtype="int")

    # build unlabeled
    labeled_texts, labeled_labels, unlabeled_texts, unlabeled_labels, all_texts, groundtruth_labels, \
        unlabeled_idx, labeled_idx = build_unlabeled(train_texts, train_labels, num_labeled=num_labeled)

    # strings ---> tokens
    labeled_processed, tokenizer = tokenize_data(labeled_texts, token_type)
    unlabeled_processed, tokenizer = tokenize_data(unlabeled_texts, token_type)
    val_processed, tokenizer = tokenize_data(val_texts, token_type)
    all_processed, tokenizer = tokenize_data(all_texts, token_type)

    # build vocab using labeled + unlabeled
    if token_type == "gru":
        token2id, id2token = build_vocab(all_processed, max_vocab=10000)
    elif token_type == "bert":
        token2id = None
        id2token = None

    # tokens --- > indices
    labeled_indices, labeled_labels = transform(labeled_processed, labeled_labels, token_type, tokenizer, token2id)
    unlabeled_indices, unlabeled_labels = transform(unlabeled_processed, unlabeled_labels, token_type, tokenizer, token2id)
    val_indices, val_labels = transform(val_processed, val_labels, token_type, tokenizer, token2id)
    all_indices, groundtruth_labels = transform(all_processed, groundtruth_labels, token_type, tokenizer, token2id)

    # build dataloaders
    if token_type == "gru":
        BATCH_SIZE = 32
        max_sent_length = 128
    elif token_type == "bert":
        BATCH_SIZE = 16
        max_sent_length = 512

    print("Creating dataloaders......")
    w_list = [1. for i in range(len(labeled_labels))]
    c_list = [1., 1.]
    train_dataset = ReviewDataset(labeled_indices, labeled_labels, w_list, c_list, max_sent_length)
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=BATCH_SIZE,
                              collate_fn=train_dataset.spam_collate_func,
                              shuffle=False)

    w_list = [1. for i in range(len(val_labels))]
    c_list = [1., 1.]
    val_dataset = ReviewDataset(val_indices, val_labels, w_list, c_list, train_dataset.max_sent_length)
    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=BATCH_SIZE,
                            collate_fn=train_dataset.spam_collate_func,
                            shuffle=False)

    w_list = [1. for i in range(len(groundtruth_labels))]
    c_list = [1., 1.]
    groundtruth_dataset = ReviewDataset(all_indices, groundtruth_labels, w_list, c_list, train_dataset.max_sent_length)
    groundtruth_loader = DataLoader(dataset=groundtruth_dataset,
                                    batch_size=BATCH_SIZE,
                                    collate_fn=groundtruth_dataset.spam_collate_func,
                                    shuffle=False)
    print("Building dataloaders done!")

    d = {
        "unlabeled_idx": unlabeled_idx,
        "labeled_idx": labeled_idx,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "groundtruth_loader": groundtruth_loader,
        "token2id": token2id,
        "id2token": id2token,
        "all_indices": all_indices,  # all 50k reviews in indices
        "groundtruth_labels": groundtruth_labels
    }

    return d
