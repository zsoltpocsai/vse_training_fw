"""Checks for missing whales in train dataset.

In case of identification, the test dataset should not contain any whale 
which is not present in the train dataset. If this happens, k-NN 
identification has no chance to give a proper identification label to unknown
whales.
"""
import sys, os, argparse, csv
sys.path.append(os.getcwd())
from utils import data_map


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("train_set_dir")
    parser.add_argument("test_set_dir")

    args = parser.parse_args()

    train_images_dir = args.train_set_dir + "/images"
    train_csv = args.train_set_dir + "/labels.csv"
    test_images_dir = args.test_set_dir + "/images"
    test_csv = args.test_set_dir + "/labels.csv"

    whales_in_train = data_map.csv2lblmap(train_csv)
    whales_in_train = list(whales_in_train.keys())

    whales_in_test = data_map.csv2lblmap(test_csv)
    whales_in_test = list(whales_in_test.keys())

    missing_from_train = 0

    for test_whale in whales_in_test:
        if test_whale not in whales_in_train:
            missing_from_train += 1

    print("Number of whales not found in train set: {}".format(missing_from_train))
