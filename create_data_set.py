import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import math
import re


import os
import random

def find_barcode(path):
    barcode_regex = r"barcode_([ACGT]+)\.csv"
    match = re.search(barcode_regex, path)
    barcode = match.group(1)
    return barcode

 

def create_data_set(file_data_name, num_clusters, num_signals_per_cluster, limit = True):
    folder_path = '/home/hadasabraham/SignalCluster/data/ground_truth_clusters'  # Replace with the path to your folder

    csv_files = []
    for root, dirs, files in os.walk(folder_path):
        for file_name in files:
            if file_name.endswith('.csv'):
                csv_files.append(os.path.join(root, file_name))

    random_files = random.sample(csv_files, k=num_clusters)

    # hadas_data_set = '/home/hadasabraham/SignalCluster/data/hadas_data_set.csv'
    data_set = []


    for file_path in random_files:
        # Process the CSV file as needed
        barcode = find_barcode(file_path)
        df_cluster = pd.read_csv(file_path)
        counter = 0
        for sig in (df_cluster.itertuples()):
            data_set.append({"signal": sig.signal, "barcode": barcode})
            if limit:
                counter += 1 
                if counter == num_signals_per_cluster:
                    break

    df_data_set = pd.DataFrame.from_records(data_set)
    df_data_set.to_csv(file_data_name) 

def clean_data_all(arrays, desired_length):
    cleaned_arrays = []
    for array in arrays:
        if len(array) > desired_length:
            cleaned_array = array[:desired_length]
        else:
            padding_length = desired_length - len(array)
            cleaned_array = np.pad(array, (0, padding_length), mode='constant')
        cleaned_arrays.append(cleaned_array)
    return cleaned_arrays

def clean_data(array, desired_length):

    if len(array) > desired_length:
        array = array[:desired_length]
    else:
        padding_length = desired_length - len(array)
        array = np.pad(array, (0, padding_length), mode='constant')
    return array