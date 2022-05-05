import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


class TfIdf:

    def __init__(self, text_names, classes, preprocessed_texts, save_name):
        super().__init__()
        self.text_names = text_names
        self.classes = classes
        self.save_name = save_name
        self.tf_idf_model = TfidfVectorizer()
        self.tf_idf_vector = self.tf_idf_model.fit_transform(preprocessed_texts)

    def create_tf_idf_matrix(self):
        tf_idf_matrix = pd.DataFrame(self.tf_idf_vector.toarray(), columns=self.tf_idf_model.get_feature_names_out())
        tf_idf_matrix.insert(0, 'text names', self.text_names)
        tf_idf_matrix['class names'] = self.classes
        return tf_idf_matrix

    def get_unique_words(self):
        return self.tf_idf_model.get_feature_names_out()



