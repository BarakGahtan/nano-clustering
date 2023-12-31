import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from create_data_set import clean_data
import os
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix
import seaborn as sns



import argparse

parser = argparse.ArgumentParser(description='Nanopore Signals Modeling')
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='batch size (default: 256)')
parser.add_argument('--epochs', type=int, default=500,
                    help='upper epoch limit (default: 200)')
parser.add_argument('--ksize', type=int, default=50,
                    help='kernel size (default: 50)')
parser.add_argument('--levels', type=int, default=8,
                    help='number of levels (default: 8)')
parser.add_argument('--nhids', type=int, default=25,
                    help='number of hidden units per layer (default: 25)')
parser.add_argument('--num_classes', type=int, default=20,
                    help='number of classes in data set (default: 20)')
args = parser.parse_args()


num_classes = args.num_classes
dropout = 0.1
kernel_size = args.ksize
'number of hidden units per layer (default: 25)'
nhids = args.nhids
'# of levels (default: 8)'
levels = args.levels
channel_sizes = [nhids] * levels
input_channels = 1
batch_size = args.batch_size
learning_rate = 1e-4
epochs = args.epochs


f_name  = f'/home/barakg/nanoclustering/hadas_adir_barak_train.csv'
# create_data_set(f_name, 20, 2, False)
col_names = ['signal', 'barcode']
data = pd.read_csv(f_name, index_col=0)
# data.columns = col_names
X = data['signal']
X = X.apply(eval).apply(lambda arr: np.array(arr, dtype=np.float32))
X = X.apply(lambda x: clean_data(x,3000))

Y = data['barcode']

y_signals = list(Y)
y_signals2 = list(set(y_signals))
# Example names vector (should have 20 names)
class_number = [i for i in range(num_classes)]

seed = 211

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=seed)

# seprate to test, validation
x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.33, random_state=seed)
print(f"Train size: {x_train.shape[0]}")
print(f"Val size: {x_val.shape[0]}")
print(f"Test size: {x_test.shape[0]}")
y_train_np = np.array([class_number[y_signals2.index(x)] for x in y_train], dtype=np.int32)
y_test_np = np.array([class_number[y_signals2.index(x)] for x in y_test], dtype=np.int32)
y_val_np = np.array([class_number[y_signals2.index(x)] for x in y_val], dtype=np.int32)
# Convert the NumPy arrays to torch tensors
x_train_tensor = torch.tensor(np.stack(x_train))
y_train_tensor = torch.from_numpy(y_train_np)

x_val_tensor = torch.tensor(np.stack(x_val))
y_val_tensor = torch.from_numpy(y_val_np)
x_test_tensor = torch.tensor(np.stack(x_test))
y_test_tensor = torch.from_numpy(y_test_np)

# plt.figure()
# for c in class_number:
#     c_x_train = x_train_tensor[y_train_tensor == c]
#     plt.plot(c_x_train[0], label="class " + str(c))
# plt.legend(loc="best")
# plt.show()
# plt.close()

x_train_tensor = x_train_tensor.reshape((x_train_tensor.shape[0], 1, x_train_tensor.shape[1]))
x_test_tensor = x_test_tensor.reshape((x_test_tensor.shape[0], 1, x_test_tensor.shape[1]))
# print(x_train_tensor.shape[0])
# print(x_train_tensor.shape[1])
# print(x_train_tensor.shape[2])
# print(type(x_train_tensor))
# print(type(y_train_tensor))
# x_train_tensor.shape[1:]
# print(type(y_train_tensor[0]))
# print(y_train_tensor[0])



class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# class TCN(nn.Module):
#     def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
#         super(TCN, self).__init__()
#         self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
#         self.transformer = torch.nn.Transformer(d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048, dropout=0.1, activation='relu', custom_encoder=None, custom_decoder=None, layer_norm_eps=1e-05, batch_first=False, norm_first=False, device=None, dtype=None)
#         self.linear = nn.Linear(num_channels[-1], output_size)
class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, inputs):
        """Inputs have to have dimension (N, C_in, L_in)"""
        y1 = self.tcn(inputs)  # input should have dimension (N, C, L)
        o = self.linear(y1[:, :, -1])
        return o




# device - cpu or gpu?
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
# loss criterion
criterion = nn.CrossEntropyLoss()

# build our model and send it to the device
model = TCN(input_size = input_channels, output_size = num_classes, num_channels = channel_sizes, kernel_size = kernel_size, dropout = dropout).to(device) # no need for parameters as we alredy defined them in the class

# optimizer - SGD, Adam, RMSProp...

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# Define custom Dataset for the training data
class MyDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, index):
        x = self.x_data[index]
        y = self.y_data[index]
        return x, y

# Create separate datasets for training data and labels
train_dataset = MyDataset(x_train_tensor, y_train_tensor)
test_dataset = MyDataset(x_test_tensor, y_test_tensor)

# dataloaders - creating batches and shuffling the data
trainloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True)
testloader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False)


# function to calcualte accuracy of the model
def calculate_accuracy(model, dataloader, device):
    model.eval() # put in evaluation mode,  turn of DropOut, BatchNorm uses learned statistics
    total_correct = 0
    total_signals = 0
    # confusion_matrix = np.zeros([num_classes,num_classes], int)
    with torch.no_grad():
        for data in dataloader:
            signals, labels = data
            signals = signals.to(device)
            labels = labels.cpu()
            outputs = model(signals)
            _, predicted = torch.max(outputs.data, 1)
            predicted = predicted.cpu()
            total_signals += labels.size(0)
            total_correct += (predicted == labels).sum().item()
            # for i, l in enumerate(labels):
            #     confusion_matrix[l.item(), predicted[i].item()] += 1
            cm = confusion_matrix(labels.numpy(), predicted.numpy())


    model_accuracy = total_correct / total_signals * 100
    return model_accuracy, cm


import time
best_accuracy = 0.0  # Initialize the best accuracy
lossvsepoch =[]
meanloss =[]
# training loop
for epoch in range(1, epochs + 1):
    model.train()  # put in training mode, turn on DropOut, BatchNorm uses batch's statistics
    running_loss = 0.0
    epoch_time = time.time()
    for i, data in enumerate(trainloader, 0):

        # get the inputs
        inputs, labels = data
        # send them to device
        inputs = inputs.to(device)
        labels = labels.to(device)
        # augmentation with `kornia` happens here inputs = aug_list(inputs)
        labels = labels.to(torch.long)
        # forward + backward + optimize
        outputs = model(inputs)  # forward pass
        loss = criterion(outputs, labels)  # calculate the loss

        # always the same 3 steps
        optimizer.zero_grad()  # zero the parameter gradients
        loss.backward()  # backpropagation
        optimizer.step()  # update parameters
        # print(f'here')
        # print statistics
        running_loss += loss.data.item()

    # Normalizing the loss by the total number of train batches
    running_loss /= len(trainloader)
    lossvsepoch.append(running_loss)
    meanloss.append(np.mean(lossvsepoch))
    # Calculate training/test set accuracy of the existing model
    train_accuracy, _ = calculate_accuracy(model, trainloader, device)
    test_accuracy, _ = calculate_accuracy(model, testloader, device)
    log = "Epoch: {} | Loss: {:.4f} | Training accuracy: {:.3f}% | Test accuracy: {:.3f}% | ".format(epoch, running_loss, train_accuracy, test_accuracy)
    # with f-strings
    # log = f"Epoch: {epoch} | Loss: {running_loss:.4f} | Training accuracy: {train_accuracy:.3f}% | Test accuracy: {test_accuracy:.3f}% |"
    epoch_time = time.time() - epoch_time
    log += "Epoch Time: {:.2f} secs".format(epoch_time)
    # with f-strings
    # log += f"Epoch Time: {epoch_time:.2f} secs"
    print(log)

    # Compare and save the model if accuracy is better
    if test_accuracy > best_accuracy:
        print('==> Saving model ...')
        state = {
            'net': model.state_dict(),
            'epoch': epoch,
            'accuracy': test_accuracy,
        }
        if not os.path.isdir('checkpoints'):
            os.mkdir('checkpoints')
        torch.save(state, f'./checkpoints/best_model_of_tcn_{batch_size}batch_{kernel_size}ker_{levels}l_{nhids}hid.pth')
        best_accuracy = test_accuracy
print('==> Finished Training ...')

# load model, calculate accuracy and confusion matrix
model = TCN(input_size = input_channels, output_size = num_classes, num_channels = channel_sizes, kernel_size = kernel_size, dropout = dropout).to(device)
state = torch.load(f'./checkpoints/best_model_of_tcn_{batch_size}batch_{kernel_size}ker_{levels}l_{nhids}hid.pth', map_location=device)
model.load_state_dict(state['net'])
# note: `map_location` is necessary if you trained on the GPU and want to run inference on the CPU

test_accuracy, cm = calculate_accuracy(model, testloader, device)
print("test accuracy: {:.3f}%".format(test_accuracy))


print(f'loss vs eppoch : {lossvsepoch}')
print(f'meanloss vs eppoch : {meanloss}')

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=[f'class {i}' for i in range (1,num_classes + 1)],
            yticklabels=[f'class {i}' for i in range (1,num_classes + 1)])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()
plt.savefig(f'confusion_matrix_of_tcn_{batch_size}batch_{kernel_size}ker_{levels}l_{nhids}hid')

#plot loss vs epochs
plt.figure(figsize=(40, 8))
plt.plot(lossvsepoch)
plt.title("Loss as a function of epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss [%]")
plt.show()
plt.savefig(f'lossvsepoch_of_tcn_{batch_size}batch_{kernel_size}ker_{levels}l_{nhids}hid')

# plot meanloss vs epochs
plt.figure(figsize=(40, 8))
plt.plot(meanloss)
plt.title("Mean loss as a function of epochs")
plt.xlabel("Epochs")
plt.ylabel("Mean loss [%]")
plt.show()
plt.savefig(f'meanlossvsepoch_of_tcn_{batch_size}batch_{kernel_size}ker_{levels}l_{nhids}hid')

