import os
import sys
import pathlib
import pandas as pd
import Operations as re
import Preprocess as pre
import TrainTest as tr
import Performance as pr

root_path = pathlib.Path(__file__).parent.resolve()

train_path = os.path.join(root_path, 'train')

test_path = os.path.join(root_path, 'test')

new_path_train = os.path.join(root_path, 'preprocessed_texts_train')

process = pre.Preprocess('localhost:6789')


if __name__ == '__main__':

    train = tr.Train(train_path, new_path_train, process, "tf-idf_train.csv")

    train.create_preprocessed_files()

    attribute_frame = pd.read_csv(os.path.join(root_path, "tf-idf_train.csv"))

    attribute_names = attribute_frame[attribute_frame.columns[~attribute_frame.columns.isin(['Unnamed: 0'])]]

    attribute_names = attribute_names.columns

    test = tr.Test(test_path, None, process, "tf-idf_test.csv", attribute_names)

    test.create_preprocessed_files()

    performance = pr.Performance(os.path.join(root_path, "tf-idf_test.csv"), os.path.join(root_path, "tf-idf_train.csv"))

    performance.naive_bayes()















