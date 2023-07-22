import torch
from torch import nn, optim
from sklearn import metrics
# import umap
from sklearn import mixture
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.manifold import Isomap
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn import preprocessing
from create_data_set import create_data_set, clean_data



# from sklearn.utils.linear_assignment_ import linear_assignment

import numpy as np
def cluster_accuracy(y_true, y_pred):
    _, ind, w = best_cluster_fit(y_true, y_pred)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

# def best_cluster_fit(y_true, y_pred):
#     y_true = y_true.astype(np.int64)
#     D = max(y_pred.max(), y_true.max()) + 1
#     w = np.zeros((D, D), dtype=np.int64)
#     for i in range(y_pred.size):
#         w[y_pred[i], y_true[i]] += 1
#
#     ind = linear_assignment(w.max() - w)
#     best_fit = []
#     for i in range(y_pred.size):
#         for j in range(len(ind)):
#             if ind[j][0] == y_pred[i]:
#                 best_fit.append(ind[j][1])
#     return best_fit, ind, w

input_dim = 3000
cluster_count = 20

class Autoencoder(nn.Module):
    def __init__(self, dims, act='relu'):
        super(Autoencoder, self).__init__()
        self.dims = dims
        self.act = act

        # Activation function
        if act == 'relu':
            self.act = nn.ReLU()
        else:
            raise ValueError('Invalid activation function.')
        # Encoder layers
        self.encoder = nn.ModuleList()
        for i in range(len(dims) - 1):
            self.encoder.append(nn.Linear(dims[i], dims[i + 1]))
        # Decoder layers
        self.decoder = nn.ModuleList()
        for i in range(len(dims) - 1, 0, -1):
            self.decoder.append(nn.Linear(dims[i], dims[i - 1]))
        # Final decoder layer
        self.final_decoder = nn.Linear(dims[1], dims[0])

    def forward(self, x):
        h = x
        for i in range(len(self.dims) - 1):
            h = self.act(self.encoder[i](h))
        for i in range(len(self.dims) - 2, 0, -1):
            h = self.act(self.decoder[i](h))
        return self.final_decoder(h)

def autoencoder_pretraining(train_dataset, val_dataset, test_dataset):
    dims = [[1,input_dim], 500, 500, 2000, cluster_count]
    model = Autoencoder(dims)
    learning_rate = 1e-3
    num_epochs = 1000
    batch_size = 256
    criterion = nn.MSELoss()  # target is the input itself
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # Data loader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    best_val_loss = float('inf')
    weights_file = 'best_weights.pth'
    for epoch in range(num_epochs):
        epoch_losses = []
        model.train()
        for features, targets in train_loader:
            features = features.to(device)
            features = features.float()
            targets = targets.to(device)
            outputs = model(features)
            loss = criterion(outputs.view(-1), targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        
        if epoch % 50 == 0:
            print(f'epoch: {epoch} loss: {np.mean(epoch_losses)}')

        # Validate the model
        model.eval()  # switch to evaluation mode
        val_loss = 0
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                outputs = model(data)
                val_loss += criterion(outputs, data).item()
        val_loss /= len(val_loader)  # average loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), weights_file)
        if epoch % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}')


    # Test the model
    model.eval()
    test_loss = 0
    model.load_state_dict(torch.load(weights_file))# Load the best weights
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            outputs = model(data)
            test_loss += criterion(outputs, data).item()
    test_loss /= len(test_loader)
    print(f'Test Loss: {test_loss:.4f}')

def eval_other_methods(x, y, names=None):
    gmm = mixture.GaussianMixture(covariance_type='full', n_components= cluster_count, random_state=0)
    gmm.fit(x)
    y_pred_prob = gmm.predict_proba(x)
    y_pred = y_pred_prob.argmax(1)
    acc = np.round(cluster_accuracy(y, y_pred), 5)
    nmi = np.round(metrics.normalized_mutual_info_score(y, y_pred), 5)
    ari = np.round(metrics.adjusted_rand_score(y, y_pred), 5)
    print(" | GMM clustering on raw data")
    print('=' * 80)
    print(acc)
    print(nmi)
    print(ari)
    print('=' * 80)

    y_pred = KMeans(n_clusters=cluster_count,random_state=0).fit_predict(x)
    acc = np.round(cluster_accuracy(y, y_pred), 5)
    nmi = np.round(metrics.normalized_mutual_info_score(y, y_pred), 5)
    ari = np.round(metrics.adjusted_rand_score(y, y_pred), 5)
    print(" | K-Means clustering on raw data")
    print('=' * 80)
    print(acc)
    print(nmi)
    print(ari)
    print('=' * 80)

    sc = SpectralClustering(n_clusters= cluster_count , random_state=0, affinity='nearest_neighbors')
    y_pred = sc.fit_predict(x)
    acc = np.round(cluster_accuracy(y, y_pred), 5)
    nmi = np.round(metrics.normalized_mutual_info_score(y, y_pred), 5)
    ari = np.round(metrics.adjusted_rand_score(y, y_pred), 5)
    print("Spectral Clustering on raw data")
    print('=' * 80)
    print(acc)
    print(nmi)
    print(ari)
    print('=' * 80)

    md = float("0.00")
    hle = umap.UMAP(random_state=0, metric='euclidean', n_components=cluster_count, n_neighbors=20, min_dist=md)\
        .fit_transform(x)

    gmm = mixture.GaussianMixture(
        covariance_type='full',
        n_components=cluster_count,
        random_state=0)

    gmm.fit(hle)
    y_pred_prob = gmm.predict_proba(hle)
    y_pred = y_pred_prob.argmax(1)
    acc = np.round(cluster_accuracy(y, y_pred), 5)
    nmi = np.round(metrics.normalized_mutual_info_score(y, y_pred), 5)
    ari = np.round(metrics.adjusted_rand_score(y, y_pred), 5)
    print('=' * 80)
    print(acc)
    print(nmi)
    print(ari)
    print('=' * 80)

if __name__ == "__main__":
    f_name  = f'/home/hadasabraham/SignalCluster/data/datasets/hadas_adir_barak_train.csv'
    # create_data_set(f_name, 20, 2, False)
    col_names = ['signal', 'barcode']
    data = pd.read_csv(f_name, index_col=0)
    # data.columns = col_names
    X = data['signal']
    X = X.apply(eval).apply(np.array)
    X = X.apply(lambda x: clean_data(x,3000))
    Y = data['barcode']
    y_signals = list(Y)
    y_signals2 = list(set(y_signals))
    # Example names vector (should have 20 names)
    class_number = [i for i in range(20)]
    
    seed = 211

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=seed)

    # seprate to test, validation 
    x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.33, random_state=seed)
    print(f"Train size: {x_train.shape[0]}")
    print(f"Val size: {x_val.shape[0]}")
    print(f"Test size: {x_test.shape[0]}")

    # # pre-processing and converting labels to integers
    # x_train_prep = preprocessing.scale(x_train)
    # x_test_prep = preprocessing.scale(x_test)
    # x_val_prep = preprocessing.scale(x_val)


    y_train_np = np.array([class_number[y_signals2.index(x)] for x in y_train]).astype(np.int)
    y_test_np = np.array([class_number[y_signals2.index(x)] for x in y_test]).astype(np.int)
    y_val_np = np.array([class_number[y_signals2.index(x)] for x in y_val]).astype(np.int)


    # Convert the NumPy arrays to torch tensors
    x_train_tensor = torch.tensor(np.stack(x_train))
    y_train_tensor = torch.from_numpy(y_train_np).float()

    x_val_tensor = torch.tensor(np.stack(x_val))
    y_val_tensor = torch.from_numpy(y_val_np).float()
    x_test_tensor = torch.tensor(np.stack(x_test))
    y_test_tensor = torch.from_numpy(y_test_np).float()
    # Create the TensorDataset
    train_dataset = torch.utils.data.TensorDataset(x_train_tensor, y_train_tensor)
    val_dataset = torch.utils.data.TensorDataset(x_val_tensor, y_val_tensor)
    test_dataset = torch.utils.data.TensorDataset(x_test_tensor, y_test_tensor)

    autoencoder_pretraining(train_dataset,val_dataset,test_dataset)

