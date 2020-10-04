import torch
import numpy as np
import torchvision
from PIL import Image
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import glob
import os 
import random 
import pandas as pd
import time
from DBN import DBN 

import numpy

path = 'Data/'
path_ls = [f for f in glob.glob(path + "**/*.jpg_color_edh_entropy", recursive=True)]
print(len(path_ls))
random.shuffle(path_ls) 

lab = [ [1,0,0,0,0], [0,1,0,0,0], [0,0,1,0,0], [0,0,0,1,0], [0,0,0,0,1] ]

data_ls = []
label_ls = []
tens_x, tens_y = [], []
# in somestring
for path in path_ls:
    if "forest" in path:
        i = 1
        label_ls.append(lab[0])
    elif "highway" in path:
        i = 2
        label_ls.append(lab[1])
    elif "insidecity" in path:
        i = 3
        label_ls.append(lab[2])
    elif "mountain" in path:
        i = 4
        label_ls.append(lab[3])
    else:
        i = 5
        label_ls.append(lab[4])
    img = np.loadtxt(path)
    img_flt = img.flatten()
    min_np, max_np = np.min(img.flatten()),np.max(img.flatten())
    img_stn = img_flt / max_np
    # img_stn = (img_flt - min_np) / (max_np - min_np)
    # min_np, max_np = np.min(img_stn.flatten()),np.max(img_stn.flatten())
    # if min_np < 0 or max_np > 1:
    #   print(min_np, max_np)
    try_ls = list(img.flatten())
    tens_x.append(try_ls.copy())
    tens_y.append(i)
    try_ls.append(i)
    data_ls.append(try_ls)

data_np = np.array(data_ls)
print(data_np.shape)
tens_x = np.array(tens_x)
tens_y = np.array(tens_y)
print("1",tens_x[0].shape)
print("2",data_np[2].shape)

# optimizer = optim.Adam(model.parameters(), lr = lr)
# model = model.to(device)

class data_generator(Dataset):
    def __init__(self, image_ls):
        self.image_ls = image_ls

    def __len__(self):
        return len(self.image_ls)

    def __getitem__(self, index):
        # label = [0,0,0,0,0,0,0]
        im = np.asarray(self.image_ls[index][:-1]).astype('float')
        im = torch.FloatTensor(im)
        label = self.image_ls[index][-1]
        # label = torch.FloatTensor(np.array(label).astype('float'))
        return  im,int(label)

batch_size = 4
device = "cpu"
EPOCHS = 2
lr = 0.5

visible_units = 828
hidden_units = [700 , 550, 300]
k = 1
learning_rate_decay = False
xavier_init = False
increase_to_cd_k = False
use_gpu = False

learning_rate = 0.1
iterations = 100

rbm = DBN(visible_units,hidden_units,k ,lr,learning_rate_decay,xavier_init,
                increase_to_cd_k,use_gpu)

# rbm_mnist.train(train_loader , EPOCHS,batch_size)
rbm.train_static(tens_x,tens_y,num_epochs=EPOCHS,batch_size=batch_size)
torch.save(rbm.state_dict(), f'wts/{EPOCHS}.pt')
# rbm.load_state_dict('wts/'+str(EPOCHS)+'.pt')

l = data_np.shape[0]
tr = int(0.8*l)
valid_np = data_np[tr:]
data_np = data_np[:tr]
print("-----------------------------")
print(data_np.shape)
print(valid_np.shape)

train_ls = []
for i in range(len(data_np)): 
    img = data_np[i]
    l = img[-1]
    img = img[:-1]
    reconstructed_img = torch.FloatTensor(np.array(img))
    _,reconstructed_img = rbm.reconstruct(reconstructed_img)
    reconstructed_img = reconstructed_img.squeeze()
    reconstructed_img = list(reconstructed_img)
    reconstructed_img.append(l)
    train_ls.append(reconstructed_img)

valid_ls = []
for i in range(len(valid_np)): 
    img = valid_np[i]
    l = img[-1]
    img = img[:-1]
    reconstructed_img = torch.FloatTensor(np.array(img))
    _,reconstructed_img = rbm.reconstruct(reconstructed_img)
    reconstructed_img = reconstructed_img.squeeze()
    reconstructed_img = list(reconstructed_img)
    reconstructed_img.append(l)
    valid_ls.append(reconstructed_img)

train_flow = data_generator(train_ls)
test_flow = data_generator(valid_ls)
train_iterator = DataLoader(train_flow, batch_size=batch_size, shuffle=True)
test_iterator = DataLoader(test_flow, batch_size=batch_size, shuffle=True)

model = nn.Sequential()
model.add_module('classifier', nn.Sequential(nn.Linear(300, 6)))

crit = nn.CrossEntropyLoss()
crit = crit.to(device)
optimizer = optim.SGD(model.parameters(), lr = learning_rate)
model = model.to(device)
print("model with classifier",model)

def calculate_accuracy(y_pred, y):
    top_pred = y_pred.argmax(1, keepdim = True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc

tr_ls, vl_ls, tr_acc, vl_acc = [],[],[],[]

for epoch in range(iterations):
    model.train()

    epoch_loss, epoch_test_loss = 0., 0.
    epoch_acc, epoch_test_acc = 0, 0
    for data in train_iterator:
        # get the inputs
        inputs, labels = data 
        inputs, labels = inputs.view(-1, 300).to(device), labels
        model.zero_grad()  # zeroes the gradient buffers of all parameters
        outputs = model(inputs) # forward 
        # print(outputs, labels)
        loss = crit(outputs, labels) # calculate loss
        # print(loss)
        acc = calculate_accuracy(outputs, labels)
        loss.backward() #  backpropagate the loss
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

        correct = 0
        total = 0
        
    model.eval()
    with torch.no_grad():
        for data in test_iterator:
            inputs, labels = data
            inputs, labels = inputs.view(-1, 300).to(device), labels
            outputs = model(inputs)
            loss = crit(outputs, labels) 
            acc = calculate_accuracy(outputs, labels)
            epoch_test_acc += acc.item()
            epoch_test_loss += loss.item()

    print(f'\tTrain Loss: {epoch_loss / len(train_iterator):.3f} | Train Acc: {(epoch_acc / len(train_iterator))*100:.2f}%')
    print(f'\t Val. Loss: {(epoch_test_loss / len(test_iterator)):.3f} |  Val. Acc: {(epoch_test_acc / len(test_iterator))*100:.2f}%')
    tr_ls.append(epoch_loss / len(train_iterator))
    vl_ls.append(epoch_test_loss / len(test_iterator))
    tr_acc.append(epoch_acc / len(train_iterator))
    vl_acc.append(epoch_test_acc / len(test_iterator))

N = np.arange(0, len(tr_ls))
plt.figure()
plt.plot(N, tr_ls, label = "train_loss")
plt.plot(N, vl_ls, label = "val_loss")
plt.title("Training Loss and Validation Loss [Epoch {}]".format(epoch))
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.ylim((-1,3))
plt.legend()
plt.savefig('wts/LossEpoch-{}.png'.format(epoch))
plt.close()
plt.figure()
plt.plot(N, tr_acc, label = "train_acc")
plt.plot(N, vl_acc, label = "val_acc")
plt.title("Training and Validation Accuracy [Epoch {}]".format(epoch))
plt.xlabel("Epoch #")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig('wts/AccuracyEpoch-{}.png'.format(epoch))
plt.close()
