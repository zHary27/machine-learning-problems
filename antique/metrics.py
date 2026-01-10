# This is an adaptation of the real script in order to work in this workspace.

import pandas as pd
from sklearn.metrics import accuracy_score
import os

def load_labels():
    base_path = os.path.dirname(os.path.abspath(__file__))
    label_path = os.path.join(base_path, 'data', 'label.csv')

    return pd.read_csv(label_path)

def get_accuracy_test_set(predictions):
    df_labels = load_labels()

    return accuracy_score(df_labels['testing_label'].values, predictions)

def get_accuracy_validation_set(predictions):
    df_labels = load_labels()

    return accuracy_score(df_labels['validation_label'].values, predictions)
    
