import os
import sys
import pandas as pd
import numpy as np
import pathlib


def get_list(path):
    words = []
    for word in list(map(lambda x: x.replace("\n", "").encode('cp1254'), open(path, encoding='cp1254').readlines())):
        words.append(word.decode('utf-8'))
    return words


def define_text_paths(class_names, path):
    text_paths = []
    for name in class_names:
        class_path = os.path.join(path, name)
        for text_name in os.listdir(class_path):
            text_paths.append(os.path.join(class_path, text_name))
    return text_paths


def get_text(text_paths):
    return open(text_paths, encoding="iso-8859-9").read()


def write_files(path, text):
    with open(path, 'w') as file:
        file.write(text)


def crate_test_data_frame(attribute_names, tf_idf_frame):
    remove = [y for y in tf_idf_frame.columns if y not in attribute_names]
    add = [x for x in attribute_names if x not in tf_idf_frame.columns]
    tf_idf_frame = tf_idf_frame[tf_idf_frame.columns.difference(remove)]
    zeros = pd.DataFrame(np.zeros((len(tf_idf_frame.index), len(add))))
    zeros.columns = add
    tf_idf_frame = pd.concat([tf_idf_frame, zeros], axis=1)
    tf_idf_frame = tf_idf_frame.reindex(columns=attribute_names)
    return tf_idf_frame


def create_report_csv(report, classes):
    arr = []
    values = []
    columns = []
    results = []
    for i in range(len(classes)):
        arr.append([x for x in report[i+2][2:] if x != ''])
    for i in range(len(classes)):
        values.append(arr[i][1:4])
        columns.append(arr[i][:1][0])
    values.append([x for x in report[-3] if x != ''][2:5])
    columns = sorted(columns)
    columns.append('Average')
    values.insert(0, ['Precision', 'Recall', 'F-Score'])
    columns.insert(0, ' ')
    results = np.transpose(values)
    data_frame = pd.DataFrame(results, columns=columns)
    data_frame.to_csv(os.path.join(pathlib.Path(__file__).parent.resolve(), "Report.csv"))
























