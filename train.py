from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from argparse import ArgumentParser
from model import *
import tensorflow as tf
import numpy as np
from helper import showClassTable, maybeExtract
import os

from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt

number_of_band = {'Indian_pines': 2, 'Salinas': 2, 'KSC': 2, 'Botswana': 1}

# get_available_gpus()
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'

# GPU_DEVICE_IDX = '1'
model_directory = os.path.join(os.getcwd(), 'Trained_model/')

parser = ArgumentParser()
parser.add_argument('--data', type=str, default='Indian_pines', help='Indian_pines or Salinas or KSC')
parser.add_argument('--epoch', type=int, default=650, help='Epochs')
parser.add_argument('--batch_size', type=int, default=50, help='Mini batch at training')
parser.add_argument('--patch_size', type=int, default=5)
parser.add_argument('--device', type=str, default='cpu')

parser = parser.parse_args()
device = torch.device(parser.device if torch.cuda.is_available() else parser.device)

TRAIN, VALIDATION, TEST = maybeExtract(parser.data, parser.patch_size)
# Extract data and label from MATLAB file


training_data, training_label = torch.from_numpy(TRAIN[0]), torch.from_numpy(TRAIN[1])
validation_data, validation_label = torch.from_numpy(VALIDATION[0]), torch.from_numpy(VALIDATION[1])
test_data, test_label = torch.from_numpy(TEST[0]), torch.from_numpy(TEST[1])

test_label =torch.squeeze(test_label)


print('training_data shape: ' + str(training_data.shape))
print('training_label shape: ' + str(training_label.shape) + '\n')
print('validation_data shape: ' + str(validation_data.shape))
print('validation_label shape: ' + str(validation_label.shape) + '\n')
print('test_data shape: ' + str(test_data.shape))
print('test_label shape:' + str(test_label.shape) + '\n')

c = torch.max(training_label)+1
c = c.numpy()

EPOCH = 50
CLASS = 11
BATCH_SIZE = 50
LR = 0.0005

TRAIN_SIZE = len(training_data)

class SpecPatConv(nn.Module):
    def __init__(self, CLASS):
        super(SpecPatConv, self).__init__()
        self.conv2d_1 = nn.Conv2d(200,200,1) #in_channel/out_channel/fiter_size
        self.fc1 = nn.Linear(200*5*5, 120)
        self.fc2 = nn.Linear(120,11)

    def forward(self, x):
        x = F.relu(self.conv2d_1(x))
        x = x.view(-1, 200*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

model = SpecPatConv(CLASS).to(device)
model = model
print(model)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

from matplotlib import cm
try: from sklearn.manifold import TSNE; HAS_SK = True
except: HAS_SK = False; print('Please install sklearn for layer visualization')


def plot_with_labels(lowDWeights, labels):
    plt.cla()
    X, Y = lowDWeights[:, 0], lowDWeights[:, 1]
    for x, y, s in zip(X, Y, labels):
        c = cm.rainbow(int(255 * s / 11));
        plt.text(x, y, s, backgroundcolor=c, fontsize=9)

    plt.xlim(X.min(), X.max())
    plt.ylim(Y.min(), Y.max())
    plt.title('Visualize last layer')
    plt.show()
    plt.pause(0.01)


def train(EPOCH=EPOCH):
    for epoch in range(EPOCH):
        for x in range(int(TRAIN_SIZE / BATCH_SIZE) + 1):

            batches = training_data[x * BATCH_SIZE: (x + 1) * BATCH_SIZE]
            labels = training_label[x * BATCH_SIZE: (x + 1) * BATCH_SIZE]

            optimizer.zero_grad()

            batches = batches.to(device)
            labels = labels.to(device)

            outputs = model(batches)

            loss = criterion(outputs, torch.max(labels, 1)[1])

            loss.backward()
            optimizer.step()

        print('Epoch ' + str(epoch + 1) + ' Loss:' + str(loss.item()) )

def test(isTraining=False):

    class_correct = list(0. for i in range(11))
    class_total = list(0. for i in range(11))
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0

        for test_batch, label in zip(test_data[:100], test_label[:100]):

            test_batch = test_batch.unsqueeze(0)
            test_batch = test_batch.float()
            label = label.long()

            test_batch = test_batch.to(device)
            label = label.to(device)

            output = model(test_batch)

            _, predicted = torch.max(output, 1)

            c = (predicted == label)
            print(c)
            total += label.size(0)

        print(correct)
        print(total)
        print('Test Accuracy of the model: {} %'.format(100 * correct / total))


def main():
    train()
    test()
    torch.save(model.state_dict(), './Trained_model/pyTorchModel.ckpt')


if __name__ == '__main__':
    main()