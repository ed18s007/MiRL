import numpy as np
import pandas as pd
import os 
import random 

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
# %matplotlib inline
import glob
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import time

path = 'Data/'
path_ls = [f for f in glob.glob(path + "**/*.jpg_color_edh_entropy", recursive=True)]
print(len(path_ls))
random.shuffle(path_ls) 

main_ls = []
# in somestring
for path in path_ls:
	if "forest" in path:
		i = 1
	elif "highway" in path:
		i = 2
	elif "insidecity" in path:
		i = 3
	elif "mountain" in path:
		i = 4
	else:
		i = 5
	img = np.loadtxt(path)
	img_flt = img.flatten()
	min_np, max_np = np.min(img.flatten()),np.max(img.flatten())
	img_stn = img_flt / max_np
	# img_stn = (img_flt - min_np) / (max_np - min_np)
	# min_np, max_np = np.min(img_stn.flatten()),np.max(img_stn.flatten())
	# if min_np < 0 or max_np > 1:
	# 	print(min_np, max_np)
	try_ls = list(img_stn.flatten())
	try_ls.append(i)
	main_ls.append(try_ls)
print(len(main_ls[0]) )
random.shuffle(main_ls) 

tr = int(0.8*len(main_ls))
tr_ls = main_ls[:tr]
vl_ls = main_ls[tr:]

##################################################################
########################    Dataloader   #########################
##################################################################
 
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

batch_size = 4
l = len(tr_ls)
train_flow = data_generator(tr_ls)
train_iterator = DataLoader(train_flow, batch_size=batch_size, shuffle=True)

##################################################################
############################   PCA   #############################
##################################################################
# REDUCED_DIM = 828

# pca = PCA(n_components=REDUCED_DIM)
# plt.figure()
# pca.fit(data_np)
# plt.plot(np.cumsum(pca.explained_variance_ratio_))
# plt.xlabel('Number of components')
# plt.ylabel('Cumulative explained variance')
# # plt.show()
# plt.savefig('wts/cumsum.png')
# plt.close()
# plt.figure()
# scaler = StandardScaler()
# scaler.fit(data_np)
# X_sc_train = scaler.transform(data_np)
# pca = PCA(n_components=REDUCED_DIM)
# X_pca_train = pca.fit_transform(X_sc_train)
# print("here", X_pca_train.shape)
# plt.plot(np.cumsum(pca.explained_variance_ratio_))
# plt.xlabel('Number of components')
# plt.ylabel('Cumulative explained variance')
# # plt.show()
# plt.savefig('wts/pcacumsum.png')

##################################################################
########################   AutoEncoder   #########################
##################################################################
class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(828, 200),
            # nn.ReLU())
            nn.Tanh())
        self.decoder = nn.Sequential(
            nn.Linear(200, 828),
            # nn.ReLU())
            nn.Tanh())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

device = "cpu"
EPOCHS = 50 #50
lr = 0.01
iterations = 100
learning_rate = 0.1

net = autoencoder()
print(net)
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

def train_autoencoder(model):
	tr_ls = [] 
	for epoch in range(EPOCHS):

		start_time = time.time()

		train_loss = train(model, train_iterator, optimizer, criterion, device)
		tr_ls.append(train_loss)

		end_time = time.time()
		epoch_mins, epoch_secs = epoch_time(start_time, end_time)
	  
		print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
		print(f'\tTrain Loss: {train_loss:.3f}')
	N = np.arange(0, len(tr_ls))
	plt.figure()
	plt.plot(N, tr_ls, label = "train_loss")
	plt.title("Training Loss AANN [Epoch {}]".format(epoch))
	plt.xlabel("Epoch #")
	plt.ylabel("Loss")
	plt.ylim((-1,3))
	plt.legend()
	plt.savefig('wts/LossEpoch-{}.png'.format(str(epoch)))
	torch.save(model.state_dict(), f'wts/{epoch}.pt')
print("Training AANN")
train_autoencoder(model)
model = nn.Sequential(*list(model.children())[:-1])
model = model.to(device)
print(model)



train_ls = []
for i in range(len(tr_ls)): 
    img = tr_ls[i]
    l = img[-1]
    img = img[:-1]
    reconstructed_img = torch.FloatTensor(np.array(img))
    reconstructed_img = model(reconstructed_img)
    reconstructed_img = reconstructed_img.squeeze()
    reconstructed_img = list(reconstructed_img)
    reconstructed_img.append(l)
    train_ls.append(reconstructed_img)

valid_ls = []
for i in range(len(vl_ls)): 
    img = vl_ls[i]
    l = img[-1]
    img = img[:-1]
    reconstructed_img = torch.FloatTensor(np.array(img))
    reconstructed_img = model(reconstructed_img)
    reconstructed_img = reconstructed_img.squeeze()
    reconstructed_img = list(reconstructed_img)
    reconstructed_img.append(l)
    valid_ls.append(reconstructed_img)

train_flow = data_generator(train_ls)
test_flow = data_generator(valid_ls)
train_iterator = DataLoader(train_flow, batch_size=batch_size, shuffle=True)
test_iterator = DataLoader(test_flow, batch_size=batch_size, shuffle=True)


model = nn.Sequential()
model.add_module('classifier_2', nn.Sequential(nn.Linear(200, 6)))

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

    for (x, y) in train_iterator:
        # get the inputs
        x = x.view(-1, 200).to(device)
        model.zero_grad()  
        outputs = model(x)
        loss = crit(outputs, y) 
        acc = calculate_accuracy(outputs, y)
        loss.backward() #  backpropagate the loss
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

        correct = 0
        total = 0
        
    model.eval()
    with torch.no_grad():
        for (x, y) in test_iterator:  
            x = x.view(-1, 200).to(device) 
            model.zero_grad() 
            outputs = model(x) 
            loss = crit(outputs, y) 
            acc = calculate_accuracy(outputs, y)
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
plt.savefig('wts/AANNLossEpoch-{}.png'.format(epoch))
plt.close()
plt.figure()
plt.plot(N, tr_acc, label = "train_acc")
plt.plot(N, vl_acc, label = "val_acc")
plt.title("Training and Validation Accuracy [Epoch {}]".format(epoch))
plt.xlabel("Epoch #")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig('wts/AANNAccuracyEpoch-{}.png'.format(epoch))
plt.close()

