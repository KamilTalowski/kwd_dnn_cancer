import os
import shutil
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from urllib.request import urlopen


def clean_run(model_dir='', source_data=''):
    """Usun model i dane dla nowego uruchomienia"""

    if model_dir:
        if os.path.exists(model_dir):
            print("Usuwanie: folder modelu [%s]." % model_dir)
            shutil.rmtree(model_dir)
            print("Usuwanie: folder modelu [%s]." % model_dir)

    for resource in [source_data, 'training_set.csv', 'test_set.csv']:
        if resource:
            if os.path.exists(resource):
                print("Usuwanie: dane [%s]." % resource)
                os.remove(resource)
                print("Usuwanie: Dane [%s]." % resource)


def download_data(data_file, url):
    """Pobierz dane jesli nie znajduja sie na dysku"""

    download_url = url + data_file

    if not os.path.exists(data_file):
        print(
            "%s Nie znaleziono plik bedzie pobrany:  %s."
            % (data_file, download_url))

        raw = urlopen(download_url).read()

        with open(data_file, 'wb') as f:
            f.write(raw)
            print("%s zapisano lokalnie." % data_file)


def process_source(local_data, col_names, missing_vals='', drop_cols=[]):
    """Czysczenie danych dla nieprawilowych wartosci i brakujacych kolumn."""

    dataframe = pd.read_csv(local_data, names=col_names)
    dataframe.replace(missing_vals, np.nan, inplace=True)
    dataframe.dropna(inplace=True)
    dataframe.drop(drop_cols, axis=1, inplace=True)

    return dataframe


def replace_classification_labels(dataframe, result_col='', values_to_replace=[]):
    """Zastapienie klasyfikacji przez numeryczna klasyfikacje."""

    target_labels = [x for x in range(0, len(values_to_replace))]
    dataframe[result_col].replace(values_to_replace, target_labels, inplace=True)

    return dataframe


def split_sets(dataframe_all):
    """Podzial danych w proporcjach 90:10."""

    train_set, test_set = train_test_split(dataframe_all, test_size=0.1, random_state=0)

    train_set.to_csv("training_set.csv", index=False, header=None)
    test_set.to_csv("test_set.csv", index=False, header=None)

    return load_tensor_data("training_set.csv"), load_tensor_data("test_set.csv")


def load_tensor_data(dataset):
    """Zaladowanie danych do tensorflow contrib.learn.dataset"""

    return tf.contrib.learn.datasets.base.load_csv_without_header(
                filename=dataset,
                target_dtype=np.int,
                features_dtype=np.float32,
                target_column=-1)


def get_inputs(data_set):
    """Zdefiniowanie wejscia dla tensor input_fn"""

    data = tf.constant(data_set.data)
    target = tf.constant(data_set.target)

    return data, target


def construct_net(num_features, model_dir):
    """Skonstruowanie 3 warstwowej sieci neuronowej z 10, 20, 10 jednostkami"""

    feature_cols = [tf.contrib.layers.real_valued_column("", dimension=num_features)]

    classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_cols,
                                                hidden_units=[10, 20, 10],
                                                n_classes=2,
                                                model_dir=model_dir)

    return classifier


def fit_model(model, train_data, steps):
    """Dopasuj model z nowymi danymi wejsciowymi"""

    model.fit(input_fn=lambda: get_inputs(train_data), steps=steps)
    print("\nModel trained after %s steps." % steps)


def evaluate_model(model, test_data, steps):
    """Ocen model dla nowych danych wejsciowych"""

    accuracy_score = model.evaluate(input_fn=lambda: get_inputs(test_data), steps=steps)["accuracy"]
    print("\nModel Accuracy: {0:f}\n".format(accuracy_score))


def new_samples(feature_names):
    """Dodaj nowe dane do klasyfikacji"""

    request_input = 0

    while int(request_input) not in [1, 2]:
        request_input = input(
            "Predict classification: Enter own data (1) or simulate fake data (2)?\n Enter 1 or 2: ")
    if int(request_input) == 1:
        sample = np.array([[int(input("Enter value 0-10 for %s: " % x)) for x in feature_names]], dtype=np.float32)
    else:
        sample = np.array([np.random.randint(11, size=len(feature_names))], dtype=np.float32)

        print("Data generated:")
        for i, x in enumerate(feature_names):
            print("%s: %s" % (x, i))

    return sample


def predict_class(model, binary_mappings):
    """Predict classification for new data"""

    predict_loop = 'Y'

    while predict_loop.upper() == 'Y':
        binary_prediction = list(model.predict(input_fn=lambda: new_samples(feature_names)))

        print("\nClass Prediction: %s\n" % binary_mappings[binary_prediction[0]])

        predict_loop = input("Would you like to try another prediction? Enter Y/N: ")


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.ERROR)

    model_dir = 'nn_classifier'
    cancer_data = 'breast-cancer-wisconsin.data'
    data_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/'

    clean_run(model_dir=model_dir)

    feature_names = ['clump_thickness', 'unif_cell_size', 'unif_cell_shape', 'marg_adhesion',
                     'single_epith_cell_size', 'bare_nuclei', 'bland_chrom', 'norm_nucleoli', 'mitoses']

    column_names = ['id'] + feature_names + ['class']

    download_data(cancer_data, data_url)

    cancer_df = process_source(cancer_data, column_names, missing_vals='?', drop_cols=['id'])
    replace_classification_labels(cancer_df, result_col='class', values_to_replace=[2, 4])

    train_set, test_set = split_sets(cancer_df)

    dnn_model = construct_net(num_features=9, model_dir=model_dir)
    fit_model(dnn_model, train_set, steps=2000)
    evaluate_model(dnn_model, test_set, steps=1)


predict_class(dnn_model, {0: 'benign', 1: 'malignant'})