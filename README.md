# LP-SSL-text-classification
NYU DS-GA 1012 final project; idea adopted from:

> A. Iscen, G. Tolias, Y. Avrithis, O. Chum. "Label Propagation for Deep Semi-supervised Learning", CVPR 2019

[Here](https://github.com/ahmetius/LP-DeepSSL) is their implementation on image classification tasks written following [Mean Teacher Pytorch implementation.](https://github.com/CuriousAI/mean-teacher/tree/master/pytorch) Part of our implementation of label propagation is derived from theirs.

## Requirements
- `python`
- `torch`
- `sacremoses`
- `transformers`
- `scipy`
- `pandas`
- `numpy`
- `scikit-learn`

## Data
We use [Large Movie Review Dataset v1.0](https://ai.stanford.edu/~amaas/data/sentiment/) for training and evaluation, which contains 50k labeled data. [Here](https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews/version/1) is a csv version of the same dataset.

## Main idea
Many supervised learning methods require a large amount of labeled data to achieve good accuracy, and in many tasks labeled data can be expensive to obtain (requires expensive human labor or human domain knowledge), while unlabeled data are available at a cheap cost. Thus, it is of practical interest to be able to leverage unlabeled data together with labeled data to reach performance comparable to that of supervised learning. Such methods, which we are interested in investigating, belong to semi-supervised learning. 

In particular, we are interested in applying [label propagation](https://pdfs.semanticscholar.org/8a6a/114d699824b678325766be195b0e7b564705.pdf), a graph-based semi-supervised learning technique, to NLP/NLU task with deep learning models. For graph-based methods, all data, with or without label, are considered as vertices on a graph in a `d-dimensional feature space`. Label propagation regards all labeled data as “sources”, and assigns pseudo-labels to unlabeled data based on the cluster assumption that vertices that are close on the graph should have similar labels. Since “unlabeled” data are now given labels inferred from labeled data, we can use them for further supervised learning. Label propagation​ has a good performance in other areas of deep learning, and we are interested in its performance on NLP/NLU tasks.

## Experiments
### Model specifics
#### Feature extractor
- `nn.Embedding` with `vocab_size=10002`
- Bi-directional `GRU` with pre-trained fasttext word embeddings
- `BERT`

#### Classifier
- `nn.Linear`

#### Criterion
- `nn.CrossEntropyLoss`

#### Optimizer
- `torch.optim.Adam(params)`

### Training pipeline
1. Assign a small portion (5~10%) of the training data `T` as the labeled dataset, `L = (x_1,x_2,...,x_l)`. Then remove the labels for the rest and call them the unlabeled dataset, `U = (x_{l+1},x_{l+2},...,x+{l+u})`.
2. Train a baseline model (e.x. 2-layer GRU with FC layer) on only `L` for `M` epochs, whose performance acts as a lower bound. Train a fully supervised model on `T` for `M` epochs, whose performance acts as an upper bound. 
3. Remove the FC layer from the baseline model to make it a feature extractor. Feed forward both `L` and `U` to get hidden representations `V = (v_1,v_2,...,v_{l+u})`. Do label propagation with `V` and assign/update the inferred labels of `U`.
4. Train the model initialized with previous weights on both `L` and `U` for one epoch.
5. Repeat 3 and 4 for `N` epochs. 


## Run the full training pipeline

### 0. Preprocessing data
Download [fasttext pre-trained word vectors](https://dl.fbaipublicfiles.com/fasttext/vectors-english/)
```shell
wget -P data_local https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip
```
 unzip
```
import zipfile
with zipfile.ZipFile("data_local/wiki-news-300d-1M.vec.zip", 'r') as zip_ref:
    zip_ref.extractall("data_local/")
```
and then finally 
```shell
python make_data.py --num_labeled 4250 --model_type gru
```
or 
```shell
python make_data.py --num_labeled 4250 --model_type bert
```

### 1. Train baseline (phase 1) model 
```shell
python train_baseline.py \
    --hidden_dim 32 \
    --num_epochs 10 \
    --name baseline \
    --num_layers 2 \
    --num_labeled 4250
    --model_type gru 
```
or 

```shell
python train_baseline.py \
    --hidden_dim 768 \
    --num_epochs 10 \
    --name baseline \
    --num_labeled 4250
    --model_type bert
```

### 2. Train full supervised (upper bound) model
```shell
python train_fully_supervised.py \
    --hidden_dim 32 \
    --num_epochs 10 \
    --name fully_supervised \
    --num_layers 2 \
    --num_labeled 4250
    --model_type gru 
```
or
```shell
python train_fully_supervised.py \
    --hidden_dim 768 \
    --num_epochs 10 \
    --name fully_supervised \
    --num_labeled 4250
    --model_type bert 
```
### 3. Train phase 2 model with pseudo labels
```shell
python train_phase2.py \
	--total_epochs 99 \
	--name phase2 \
	--num_labeled 4250 \
	--knn 100 \
	--phase1_model_name baseline_bert \
    --model_type bert
```

or

```shell
python train_phase2.py \
    --total_epochs 99 \
    --name phase2 \
    --num_labeled 4250 \
    --knn 100 \
    --phase1_model_name baseline_gru \
    --model_type gru
```
If successful, we should see that the performance of this model lies between that of phase 1 model and the fully-supervised model. We can also test how performance improves with more labeled data.
