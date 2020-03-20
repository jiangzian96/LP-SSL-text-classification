from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from sklearn.neighbors import NearestNeighbors
import scipy
from sklearn.preprocessing import normalize

from data_local.utils import *


# removing the last FC layer for the feature extractor
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class GRUClassifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_layers, num_classes=2, vocab_size=10002):
        super().__init__()
        self.embedding_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=0)
        self.gru = nn.GRU(input_size=embedding_dim, hidden_size=hidden_dim, batch_first=True, num_layers=num_layers)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, inputs):
        out = self.embedding_layer(inputs)
        gru_out, h_n = self.gru(out)
        # gru_out: (batch, seq_len, num_directions * hidden_size)
        out = gru_out[:, -1, :].squeeze(1)
        logits = self.fc(out)

        return logits


def create_model(args):
    embedding_dim = args.embedding_dim
    hidden_dim = args.hidden_dim
    num_layers = args.num_layers

    return GRUClassifier(embedding_dim=embedding_dim, hidden_size=hidden_dim, num_layers=num_layers)


def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    print("Evaluating model...")

    for data_batch, labels_batch, w, c in tqdm(dataloader):
        data_batch = data_batch.to(device)
        labels_batch = labels_batch.to(device)
        outputs = F.softmax(model(data_batch), dim=1)
        predicted = outputs.max(1, keepdim=True)[1]
        total += labels_batch.size(0)
        correct += predicted.eq(labels_batch.view_as(predicted)).sum().item()

    accuracy = correct / total
    print("Evaluation done! accuracy: {}".format(accuracy))

    return accuracy


def train(train_loader, val_loader, model, optimizer, criterion, device, args):
    train_loss_history = []
    val_accuracy_history = []
    best_val_acc = 0
    max_epoch = args.num_epochs
    name = args.name

    for epoch in tqdm(range(max_epoch)):
        model.train()
        for i, (data_batch, batch_labels, w, c) in enumerate(train_loader):
            preds = model(data_batch.to(device))
            w = Variable(w, requires_grad=True).to(device)
            c = Variable(c, requires_grad=True).to(device)
            minibatch_size = len(batch_labels)

            # criterion has reduction="none"
            loss = criterion(preds, batch_labels.to(device))

            # element-wise multiplications
            loss = loss * w
            loss = loss * c
            loss = loss.sum() / minibatch_size

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_loss_history.append(loss.item())

        print("loss at the end of epoch {}: ".format(epoch), loss.item())

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


def extract_features(data_loader, model_path, device):
    args = torch.load(model_path, map_location=torch.device(device))["args"]

    # build model
    feature_extractor = create_model(args).to(device)
    feature_extractor.load_state_dict(torch.load(model_path, map_location=torch.device(device))["model_state_dict"])
    feature_extractor.fc = Identity()

    feature_extractor.eval()
    res = torch.tensor([]).to(device)
    print("Extracting features......")
    for i, (data_batch, batch_labels, w, c) in enumerate(tqdm(data_loader)):
        batch_features = feature_extractor(data_batch.to(device))
        res = torch.cat((res, batch_features), 0)

    print("Extracted {} points; each {}-dimensional!".format(*res.shape))

    return res


def update_pseudoloader(all_indices, p_labels, updated_weights, updated_class_weights):
    pseudo_dataset = ReviewDataset(all_indices, p_labels, updated_weights, updated_class_weights, 128)
    pseudo_loader = torch.utils.data.DataLoader(dataset=pseudo_dataset,
                                                batch_size=32,
                                                collate_fn=pseudo_dataset.spam_collate_func,
                                                shuffle=False)
    return pseudo_loader


def f1_score(y_true, y_pred):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_pred)):
        if y_true[i] == y_pred[i] == 1:
            TP += 1
        if y_pred[i] == 1 and y_true[i] != y_pred[i]:
            FP += 1
        if y_true[i] == y_pred[i] == 0:
            TN += 1
        if y_pred[i] == 0 and y_true[i] != y_pred[i]:
            FN += 1

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * (precision * recall) / (precision + recall)

    return f1


def label_propagation(batch_features, groundtruth_labels, labeled_idx, unlabeled_idx, num_classes=2, k=100):
    print("Calculating knn......")
    X, distances, indices = calculate_knn(batch_features, k=k)

    print("Building graph......")
    W_normalized = build_graph(X, distances, indices, gamma=3, k=k)

    print("Running label propagation......")
    groundtruth_labels = np.array(groundtruth_labels)
    Z = calculate_label_distributions(W_normalized, groundtruth_labels, labeled_idx, num_classes=2, alpha=0.99)

    print("Assigning pseudo labels and weights......")
    p_labels, weights = assign_pseudo_labels(Z, groundtruth_labels, labeled_idx, num_classes=2)

    print("Assigning class weights.......")
    class_weights = [None for i in range(num_classes)]
    for i in range(num_classes):
        cur_idx = np.where(p_labels == i)[0]
        class_weights[i] = float(len(groundtruth_labels) / num_classes) / cur_idx.size

    return p_labels.tolist(), weights.tolist(), class_weights


def assign_pseudo_labels(Z, groundtruth_labels, labeled_idx, num_classes=2):
    # from https://github.com/ahmetius/LP-DeepSSL/blob/master/lp/db_semisuper.py

    Z = F.normalize(torch.tensor(Z), 1).numpy()
    Z[Z < 0] = 0
    entropy = scipy.stats.entropy(Z.T)

    # eq. 11 from paper
    weights = 1 - entropy / np.log(num_classes)
    weights = weights / np.max(weights)

    p_labels = np.argmax(Z, 1)
    p_labels[labeled_idx] = groundtruth_labels[labeled_idx]

    return p_labels, weights


def calculate_knn(batch_features, k=100):
    X = batch_features.cpu().detach().numpy()
    X = normalize(X)
    print("Calculating {} nearest neighbors for each point...".format(k))
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(X)
    distances, indices = nbrs.kneighbors(X)
    indices = indices[:, 1:]
    distances = distances[:, 1:]

    return X, distances, indices


def build_graph(X, distances, indices, gamma=3, k=100):
    # from https://github.com/ahmetius/LP-DeepSSL/blob/master/lp/db_semisuper.py

    N = X.shape[0]
    row_idx = np.arange(N)
    row_idx_rep = np.tile(row_idx, (k, 1)).T
    values = distances ** gamma
    W = scipy.sparse.csr_matrix((values.flatten('F'), (row_idx_rep.flatten('F'), indices.flatten('F'))), shape=(N, N))
    W = W + W.T
    W = W - scipy.sparse.diags(W.diagonal())
    S = W.sum(axis=1)
    S[S == 0] = 1
    D = np.array(1. / np.sqrt(S))
    D = scipy.sparse.diags(D.reshape(-1))
    W_normalized = D * W * D

    return W_normalized


def calculate_label_distributions(W, all_labels, labeled_idx, num_classes=2, alpha=0.99, maxiter=20):
    # from https://github.com/ahmetius/LP-DeepSSL/blob/master/lp/db_semisuper.py

    N = W.shape[0]
    Z = np.zeros((N, num_classes))  # label_distributions
    A = scipy.sparse.eye(N) - alpha * W

    for i in range(num_classes):
        cur_idx = np.asarray(labeled_idx)[np.where(np.asarray(all_labels)[labeled_idx] == i)]
        y = np.zeros((N,))
        # eq.5 from paper
        y[cur_idx] = 1.0 / cur_idx.shape[0]
        # eq.6 and 10 from paper
        f, _ = scipy.sparse.linalg.cg(A, y, tol=1e-6, maxiter=maxiter)
        Z[:, i] = f

    Z[Z < 0] = 0

    return Z
