import torch
import numpy as np
import torchvision
from PIL import Image
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import glob
import os 
import random 
import pandas as pd
import time

path = 'Data/'
path_ls = [f for f in glob.glob(path + "**/*.jpg_color_edh_entropy", recursive=True)]
print(len(path_ls))
random.shuffle(path_ls) 

lab = [ [1,0,0,0,0], [0,1,0,0,0], [0,0,1,0,0], [0,0,0,1,0], [0,0,0,0,1] ]

data_ls = []
label_ls = []
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
	# 	print(min_np, max_np)
	try_ls = list(img.flatten())
	try_ls.append(i)
	data_ls.append(try_ls)

data_np = np.array(data_ls)
label_np = np.array(label_ls)
print(data_np.shape, label_np.shape)

batch_size = 2
device = "cpu"
EPOCHS = 10
lr = 0.5
iterations = 10
learning_rate = 0.001

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
        im = (im-min(im))/(max(im)-min(im))
        im = torch.FloatTensor(im)
        label = self.image_ls[index][-1]
        # label = torch.FloatTensor(np.array(label).astype('float'))
        return  im,label

l = len(data_ls)
tr = int(0.8*l)
train_flow = data_generator(data_ls[:tr])
test_flow = data_generator(data_ls[tr:])
train_iterator = DataLoader(train_flow, batch_size=batch_size, shuffle=True)
test_iterator = DataLoader(test_flow, batch_size=batch_size, shuffle=True)


class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(828, 600),
            nn.ReLU())
            # nn.Tanh())
        self.decoder = nn.Sequential(
            nn.Linear(600, 828),
            nn.ReLU())
            # nn.Sigmoid())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


net = autoencoder()
# print(net)
model = net.to(device)

criterion = nn.MSELoss()
criterion = criterion.to(device)
optimizer = optim.SGD(model.parameters(), lr = lr)

def train(model, iterator, optimizer, criterion, device):
    epoch_loss = 0   
    model.train()
    for (x, y) in iterator:
        x = x.view(-1, 828).to(device)
        # y = y.to(device)
        optimizer.zero_grad()
        y_pred = model(x)
        loss = criterion(y_pred, x)
                
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def train_autoencoder(stacked_layer,model):
	tr_ls = [] 
	for epoch in range(EPOCHS):

		start_time = time.time()

		train_loss = train(model, train_iterator, optimizer, criterion, device)
		tr_ls.append(train_loss)

		torch.save(model.state_dict(), f'wts/{epoch}.pt')
		end_time = time.time()
		epoch_mins, epoch_secs = epoch_time(start_time, end_time)
	  
		print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
		print(f'\tTrain Loss: {train_loss:.3f}')
	N = np.arange(0, len(tr_ls))
	plt.figure()
	plt.plot(N, tr_ls, label = "train_loss")
	plt.title("Training Loss [Epoch {}]".format(epoch))
	plt.xlabel("Epoch #")
	plt.ylabel("Loss")
	plt.ylim((-1,3))
	plt.legend()
	plt.savefig('wts/LossEpoch-{}.png'.format(str(stacked_layer)+"_"+str(epoch)))
print("Training First Stacked Auto-Encoder")
train_autoencoder(1,model)

model.encoder.add_module('First_Encoder_Layer', nn.Sequential(nn.Linear(600, 300),nn.ReLU()))
model.encoder.add_module('First_Decoder_Layer', nn.Sequential(nn.Linear(300, 600),nn.ReLU()))
model = model.to(device)
# print("model",model)
print("Training Second Stacked Auto-Encoder")
train_autoencoder(2,model)

model = nn.Sequential(*list(model.children())[:-1])
model = nn.Sequential(*list(model[0].children())[:-1])
model.add_module('Second_Encoder_Layer', nn.Sequential(nn.Linear(300, 150),nn.ReLU()))
model.add_module('First_Decoder_Layer', nn.Sequential(nn.Linear(150, 300),nn.ReLU()))
model.add_module('Second_Decoder_Layer', nn.Sequential(nn.Linear(300, 600),nn.ReLU()))
model.add_module('Decoder', nn.Sequential(nn.Linear(600, 828),nn.ReLU()))
model = model.to(device)
# print("model",model)
print("Training Third Stacked Auto-Encoder")
train_autoencoder(3,model)

model = nn.Sequential(*list(model.children())[:-3])
model.add_module('classifier', nn.Sequential(nn.Linear(150, 6)))
model = model.to(device)
# print(model)


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
        inputs, labels = inputs.view(-1, 828).to(device), labels
        model.zero_grad()  # zeroes the gradient buffers of all parameters
        outputs = model(inputs) # forward 
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
            inputs, labels = inputs.view(-1, 828).to(device), labels
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
