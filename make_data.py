import argparse
import pickle

from data_local.utils import *


def main():
    # generate labeled and unlabeled data
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num_labeled", default=4250, type=int, help="number of labeled data")
    args = parser.parse_args()

    d = build_dataloaders(num_labeled=args.num_labeled)
    with open('data_local/processed/data_{}.pickle'.format(args.num_labeled), 'wb') as handle:
        pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
