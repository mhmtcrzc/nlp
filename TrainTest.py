import os
import sys
import shutil
import Operations as re
import TfIdf as tf
import FeatureSelection as fea
import pandas as pd
import pathlib


class Train:

    def __init__(self, train_path, new_path, pre_instance, csv_name):
        super().__init__()
        self.csv_name = csv_name
        self.pre_instance = pre_instance
        self.new_path = new_path
        try:
            self.class_names = set(os.listdir(train_path))
        except:
            print("File path not found please check your data. \nExiting the program...")
            sys.exit()
        self.text_paths = re.define_text_paths(self.class_names, train_path)
        self.preprocessed_texts = []
        self.text_names = []
        self.classes = []
        self.selected_columns = None

    def create_preprocessed_files(self):
        self.create_paths()
        for path in self.text_paths:
            temporary_text = " "
            text_name = path.split("/")[-1]
            class_name = path.split("/")[-2]
            preprocessed_text = self.pre_instance.preprocess(re.get_text(path))
            try:
                re.write_files(os.path.join(self.new_path, class_name, text_name), str(preprocessed_text))
            except:
                print("Please check file permissions. \nExiting the program...")
                sys.exit()
            self.text_names.append(text_name)
            self.preprocessed_texts.append(temporary_text.join(preprocessed_text))
            self.classes.append(class_name)
        tf_idf = tf.TfIdf(self.text_names, self.classes, self.preprocessed_texts, self.csv_name)
        feature_selection = fea.FeatureSelection(tf_idf.create_tf_idf_matrix(), self.csv_name)
        feature_selection.mutual_information()
        self.selected_columns = feature_selection.selected_columns

    def create_paths(self):
        try:
            if os.path.exists(os.path.join(self.new_path)):
                shutil.rmtree(self.new_path)
            for path in self.class_names:
                if not os.path.exists(os.path.join(self.new_path, path)):
                    os.makedirs(os.path.join(self.new_path, path))
        except:
            print("Please check file permissions. \nExiting the program...")
            sys.exit()


class Test(Train):

    def __init__(self, train_path, new_path, pre_instance, csv_name, attributes_names):
        super().__init__(train_path, new_path, pre_instance, csv_name)
        self.attribute_names = attributes_names

    def create_preprocessed_files(self):
        for path in self.text_paths:
            temporary_text = " "
            text_name = path.split("/")[-1]
            class_name = path.split("/")[-2]
            preprocessed_text = self.pre_instance.preprocess(re.get_text(path))
            self.text_names.append(text_name)
            self.preprocessed_texts.append(temporary_text.join(preprocessed_text))
            self.classes.append(class_name)
        tf_idf = tf.TfIdf(self.text_names, self.classes, self.preprocessed_texts, self.csv_name)
        tf_idf_frame = tf_idf.create_tf_idf_matrix()
        tf_idf_frame = re.crate_test_data_frame(self.attribute_names, tf_idf_frame)
        tf_idf_frame.to_csv(os.path.join(pathlib.Path(__file__).parent.resolve(), self.csv_name))

    def create_paths(self):
        pass
