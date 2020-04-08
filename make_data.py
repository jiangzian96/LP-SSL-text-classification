import argparse
import pickle

from data_local.utils import *


def main():
    # generate labeled and unlabeled data
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num_labeled", default=4250, type=int, help="number of labeled data")
    parser.add_argument("-t", "--model_type", type=str, help="type of tokenization", required=True, choices=["gru", "bert"])
    args = parser.parse_args()

    d = build_dataloaders(args)
    with open('data_local/processed/data_{}_{}.pickle'.format(args.model_type, args.num_labeled), 'wb') as handle:
        pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
