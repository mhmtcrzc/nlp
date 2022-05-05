from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report
import Operations as re
import pandas as pd
import pickle
import pathlib
import os

class Performance:

    def __init__(self, test_path, train_path):
        super().__init__()
        self.test_frame = pd.read_csv(test_path)
        self.test_frame = self.test_frame[self.test_frame.columns[~self.test_frame.columns.isin(['Unnamed: 0'])]]
        self.train_frame = pd.read_csv(train_path)
        self.train_frame = self.train_frame[self.train_frame.columns[~self.train_frame.columns.isin(['Unnamed: 0'])]]

    def naive_bayes(self):
        train_x = self.train_frame[self.train_frame.columns.difference(['text names', 'class names'])]
        train_y = self.train_frame['class names']
        test_x = self.test_frame[self.test_frame.columns.difference(['class names', 'text names'])]
        test_y = self.test_frame['class names']
        classes = frozenset(test_y)
        target_classes = list(set(test_y))
        nb = MultinomialNB()
        nb.fit(train_x, train_y)
        prediction = nb.predict(test_x)
        report = classification_report(test_y, prediction, target_names=target_classes)
        report = [x.split(' ') for x in report.split('\n')]
        re.create_report_csv(report, classes)
        merge_test_train = pd.concat([self.train_frame, self.test_frame])
        merge_test_train.to_csv((os.path.join(pathlib.Path(__file__).parent.resolve(), "TF-IDF-ALL.csv")))





