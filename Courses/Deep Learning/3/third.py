import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import random
from netVLAD import NetVLAD
# from netVLAD import EmbedNet
import numpy as np 
import cv2
import pandas as pd 
import os 
import torchvision.models as models
import torch.nn.functional as F
import torch.optim as optim
import time
import matplotlib.pyplot as plt

# img_size, fc_size = 12, 16
# img_size, fc_size = 16, 64
# img_size, fc_size = 20, 144
# img_size, fc_size = 24, 256 
img_size, fc_size = 28, 400 

EPOCHS = 100
batch_size = 8 # 12, 20
lr = 0.1 # 0.05, 0.01
fc1, fc2 = fc_size,fc_size

class data_generator(Dataset):
    def __init__(self, image_ls):
        self.image_ls = image_ls

    def __len__(self):
        return len(self.image_ls)

    def __getitem__(self, index):
        # label = [0,0,0,0,0,0,0]
        im = cv2.resize(cv2.imread(self.image_ls[index][1]),(img_size,img_size))
        im = im/255.0
        im = torch.FloatTensor(np.asarray(im).transpose(-1, 0, 1).astype('float'))
        # label[self.image_ls[index][2]-1]=1
        # label = torch.FloatTensor(np.array(label).astype('float'))
        return (im, self.image_ls[index][2]-1)
        
filename = 'train.csv'
data = pd.read_csv(filename)
train_ls = data.values.tolist()
random.shuffle(train_ls)
# train_ls = train_ls[:20]

filename = 'valid.csv'
data = pd.read_csv(filename)
valid_ls = data.values.tolist() 

filename = 'test.csv'
data = pd.read_csv(filename)
test_ls = data.values.tolist() 

print("train_ls,valid_ls length is",len(valid_ls),len(train_ls))
train_flow = data_generator(train_ls)
valid_flow = data_generator(valid_ls)
test_flow = data_generator(test_ls)
train_iterator = DataLoader(train_flow, batch_size=batch_size, shuffle=True)
valid_iterator = DataLoader(valid_flow, batch_size=batch_size, shuffle=True)
test_iterator = DataLoader(test_flow, batch_size=batch_size, shuffle=False)


class CNN(nn.Module):
	def __init__(self):
		super(CNN, self).__init__()
		self.CL1_PL1 = nn.Sequential(
			nn.Conv2d(3,4,kernel_size=3,stride=1),
			nn.ReLU(),
			nn.AvgPool2d(kernel_size=2,stride=2)
			)
		self.CL2_PL2 = nn.Sequential(
			nn.Conv2d(4,16,kernel_size=3,stride=1),
			nn.ReLU(),
			nn.AvgPool2d(kernel_size=2,stride=2)
			)

	def forward(self,x):
		out = self.CL1_PL1(x)
		out = self.CL2_PL2(out)
		return out


class EmbedNet(nn.Module):
    def __init__(self, base_model, net_vlad,dim):
        super(EmbedNet, self).__init__()
        self.base_model = base_model
        self.net_vlad = net_vlad
        self.fc1 = nn.Linear(64,fc1)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(fc1,7)
        # self.act2 = nn.ReLU()
        # self.fc3 = nn.Linear(fc1,7)

    def forward(self, x):
        x = self.base_model(x)
        embedded_x = self.net_vlad(x)
        out = self.fc1(embedded_x)
        out = self.act1(out)
        out = self.fc2(out)
        return out

cnnmodel = CNN()
print(cnnmodel)
dim = list(cnnmodel.parameters())[-1].shape[0] 
# Define model for embedding
net_vlad = NetVLAD(num_clusters=4, dim=dim, alpha=1.0)
model = EmbedNet(cnnmodel, net_vlad,dim)
print(model)

# Define loss

optimizer = optim.Adam(model.parameters(), lr = lr)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
criterion = nn.CrossEntropyLoss()
model = model.to(device)
criterion = criterion.to(device)

def calculate_accuracy(y_pred, y):
    top_pred = y_pred.argmax(1, keepdim = True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc

def train(model, iterator, optimizer, criterion, device):
    epoch_loss = 0
    epoch_acc = 0
    model.train()
    for (x, y) in iterator:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        y_pred = model(x)
        loss = criterion(y_pred, y)
        # print("loss",loss)
        acc = calculate_accuracy(y_pred, y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, criterion, device):
    epoch_loss = 0
    epoch_acc = 0
    model.eval()
    with torch.no_grad():
        for (x, y) in iterator:
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x)
            loss = criterion(y_pred, y)
            acc = calculate_accuracy(y_pred, y)
            epoch_loss += loss.item()
            epoch_acc += acc.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


best_valid_loss = float('inf')

tr_ls, vl_ls, tr_acc, vl_acc, tes_ls, tes_acc = [],[],[],[],[],[]
for epoch in range(EPOCHS):
    start_time = time.time()
    train_loss, train_acc = train(model, train_iterator, optimizer, criterion, device)
    valid_loss, valid_acc = evaluate(model, valid_iterator, criterion, device)
    test_loss, test_acc = evaluate(model, test_iterator, criterion, device)
    tr_ls.append(train_loss)
    vl_ls.append(valid_loss)
    tes_ls.append(test_loss)
    tes_acc.append(test_acc)
    tr_acc.append(train_acc)
    vl_acc.append(valid_acc)

    # if valid_loss < best_valid_loss:
    best_valid_loss = valid_loss
    torch.save(model.state_dict(), f'third/{epoch}.pt')
    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')
    print(f'\t Test. Loss: {test_loss:.3f} |  Test. Acc: {test_acc*100:.2f}%')
N = np.arange(0, len(tr_ls))
plt.figure()
plt.plot(N, tr_ls, label = "train_loss")
plt.plot(N, vl_ls, label = "val_loss")
plt.plot(N, tes_ls, label = "test_loss")
plt.title("Training Loss and Validation Loss [Epoch {}]".format(epoch))
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.ylim((-1,3))
plt.legend()
plt.savefig('third/LossEpoch-{}.png'.format(epoch))
plt.close()
plt.figure()
plt.plot(N, tr_acc, label = "train_acc")
plt.plot(N, vl_acc, label = "val_acc")
plt.plot(N, tes_acc, label = "test_acc")
plt.title("Training and Validation Accuracy [Epoch {}]".format(epoch))
plt.xlabel("Epoch #")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig('third/AccuracyEpoch-{}.png'.format(epoch))
plt.close()