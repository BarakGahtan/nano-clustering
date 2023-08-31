import torch.nn as nn
from torch.nn.utils import weight_norm

from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from create_data_set import clean_data
from dtaidistance import dtw
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import argparse
import seaborn as sns


parser = argparse.ArgumentParser(description='DTW to compare')
parser.add_argument('--with_padding', type=bool, default=True,
                    help='all signals with len 3000 (default: True)')
parser.add_argument('--num_classes', type=int, default=20,
                    help='number of classes in data set (default: 20)')
args = parser.parse_args()


num_classes = args.num_classes
with_padding = args.with_padding

# f_name = f'/home/hadasabraham/SignalCluster/data/datasets/train_100clusters.csv'
f_name  = f'/home/hadasabraham/SignalCluster/data/datasets/hadas_adir_barak_train.csv'
exp_name = 'train_20clusters'


def normalize_arrays(arrays):
    normalized_arrays = []
    kernel_size = 5
    kernel = np.ones(kernel_size) / kernel_size

    for array in arrays:
        normalized_array = array[~np.isnan(array)]
        normalized_array = np.convolve(normalized_array, kernel, mode='same')
        normalized_array = (normalized_array - np.mean(normalized_array)) / np.std(normalized_array)
        normalized_arrays.append(normalized_array)

    return normalized_arrays


def f1_score_calc(mat, mat_pred):
    print('Precision is: ', precision_score(mat, mat_pred))
    print('Recall is: ', recall_score(mat, mat_pred))
    f1score = f1_score(mat, mat_pred)
    print('F1 is: ', f1score)
    return f1score


### Thershold
def mat_with_thershold(val, df):
    D = df.copy()
    for i in range(len(D)):
        D[i] = 1 if D[i] > val else 0
    return D

def create_matrix_true_labales(input_list):
    n = len(input_list)
    matrix = [[0] * n for _ in range(n)]

    for i in range(n):
        for j in range(n):
            if input_list[i] == input_list[j]:
                matrix[i][j] = 1

    return matrix


##########################################
col_names = ['signal', 'barcode']
data = pd.read_csv(f_name, index_col=0)
# data.columns = col_names
X = data['signal']
X = X.apply(eval).apply(lambda arr: np.array(arr, dtype=np.float32))
X = X.apply(lambda x: clean_data(x,3000)) if with_padding else X

Y = data['barcode']
y_signals = list(Y)
y_signals2 = list(set(y_signals))
# Example names vector (should have 100 names)
class_number = [i for i in range(num_classes)]
y_train_np = np.array([class_number[y_signals2.index(x)] for x in Y], dtype=np.int32)

X = normalize_arrays(X)
D = [[0 for _ in range(len(X))] for _ in range(len(X))]


for i in range(len(X)):
    for j in range(len(X)):
        s = X[i]
        t = X[j]
        score = dtw.distance(s, t)
        D[i][j] = score
        D[j][i] = score
        print(score)

df = pd.DataFrame(D)
df.to_csv(f'/home/hadasabraham/nanocluser/nanocluster/data/matrix_to_compare/dtw_on_{exp_name}_with_padding:{with_padding}.csv', index=None, header=None)
D = np.reshape(D, (len(X)*len(X),1))

best_acc = 0
best_thersh = 0
for i in range(10,30,0.5):
    pred_mat = mat_with_thershold(i, D)
    true_labels = np.zeros((len(X), len(X)))
    true_labels = create_matrix_true_labales(y_train_np)
    true_labels = np.reshape(true_labels, (len(X)*len(X),1))
    accuracy = f1_score_calc(true_labels, pred_mat)
    print(f'thershold is : {i}, accuracy is {accuracy}')
    if best_acc < accuracy:
        best_acc = accuracy
        best_thersh = i
        cm = confusion_matrix(true_labels, pred_mat)
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=[f'class {i}' for i in range (1,num_classes + 1)],
                    yticklabels=[f'class {i}' for i in range (1,num_classes + 1)])
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        plt.show()
        plt.savefig(f'confusion_matrix_of_dtw_on_{exp_name}_with_padding:{with_padding}')


print(f'best accuracy is {best_acc}, with thershold {best_thersh}')




