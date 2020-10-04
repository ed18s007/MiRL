import numpy as np 
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd 
import os 

class data_generator(Dataset):
    def __init__(self, image_ls):
        self.image_ls = image_ls

    def __len__(self):
        return len(self.image_ls)

    def __getitem__(self, index):
        # label = [0,0,0,0,0,0,0]
        im = cv2.resize(cv2.imread(self.image_ls[index][1]),(224,224))
        im = torch.FloatTensor(np.asarray(im).transpose(-1, 0, 1).astype('float'))
        # label[self.image_ls[index][2]-1]=1
        # label = torch.FloatTensor(np.array(label).astype('float'))
        return (im, self.image_ls[index][2]-1)

filename = 'train.csv'
data = pd.read_csv(filename)
train_ls = data.values.tolist()
# random.shuffle(train_ls)
train_ls = train_ls[:20]

filename = 'valid.csv'
data = pd.read_csv(filename)
valid_ls = data.values.tolist() 

filename = 'test.csv'
data = pd.read_csv(filename)
test_ls = data.values.tolist() 

print("train_ls,valid_ls length is",len(valid_ls),len(train_ls))
train_flow = data_generator(train_ls)
valid_flow = data_generator(valid_ls)
# test_flow = data_generator(test_ls, batch_size = 1)
train_iterator = DataLoader(train_flow, batch_size=5, shuffle=True)
valid_iterator = DataLoader(valid_flow, batch_size=5, shuffle=True)

for i_batch, sample_batched in enumerate(train_iterator):
    print(i_batch)

import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import torch.optim as optim
import time
OUTPUT_DIM = 7
pretrained_model = models.vgg16_bn(pretrained = True)
print(pretrained_model)


IN_FEATURES = pretrained_model.classifier[-1].in_features 
final_fc = nn.Linear(IN_FEATURES, OUTPUT_DIM)
pretrained_model.classifier[-1] = final_fc
print(pretrained_model.classifier)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(pretrained_model):,} trainable parameters')

for parameter in pretrained_model.features.parameters():
    parameter.requires_grad = False

START_LR = 1e-7

optimizer = optim.Adam(pretrained_model.parameters(), lr = START_LR)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

criterion = nn.CrossEntropyLoss()

model = pretrained_model.to(device)
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

EPOCHS = 5

best_valid_loss = float('inf')

for epoch in range(EPOCHS):
    
    start_time = time.time()
    
    train_loss, train_acc = train(model, train_iterator, optimizer, criterion, device)
    valid_loss, valid_acc = evaluate(model, valid_iterator, criterion, device)
        
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'tut4-model.pt')

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
    print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')