import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, mutual_info_classif
import pathlib
import os


class FeatureSelection:

    def __init__(self, tf_idf_matrix, save_name):
        super().__init__()
        self.tf_idf_matrix = tf_idf_matrix
        self.k = 2000
        self.save_name = save_name
        self.selected_columns = None

    def mutual_information(self):
        select_k_best = SelectKBest(mutual_info_classif, k=self.k)
        x = self.tf_idf_matrix[self.tf_idf_matrix.columns[~self.tf_idf_matrix.columns.isin(['class names', 'text names'])]]
        y = self.tf_idf_matrix[self.tf_idf_matrix.columns[self.tf_idf_matrix.columns.isin(['class names'])]]
        select_k_best.fit(x, np.ravel(y))
        results = select_k_best.get_support()
        selected_columns = []
        for i in range(0, len(x.columns)):
            if results[i]:
                selected_columns.append(x.columns[i])
        new_matrix = pd.DataFrame(self.tf_idf_matrix, columns=selected_columns)
        self.selected_columns = selected_columns
        new_matrix.insert(0, 'text names', self.tf_idf_matrix['text names'])
        new_matrix['class names'] = self.tf_idf_matrix['class names']
        new_matrix = new_matrix[new_matrix.columns[~new_matrix.columns.isin(['Unnamed: 0'])]]
        new_matrix.to_csv(os.path.join(pathlib.Path(__file__).parent.resolve(), self.save_name))




