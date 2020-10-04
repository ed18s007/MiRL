import os
import pandas as pd 
import cv2
import numpy as np
print("The list of classes for group 35 is")
print(os.listdir("data"))

train_ls, valid_ls, test_ls = [], [], []
label = 1
tr, vl, ts = 0.7*60, 0.7*60+0.1*60 , 0.7*60 + 0.1*60+ 0.2*60
for path in os.listdir("data"):
    data_ls = []
    i = 0
    for file in os.listdir("data/"+path):
        item = []
        item.append("data/"+path+"/"+file)
        item.append(label)
        if i<tr:
            filename = "Aug_Data/"+path+"/"+file[:-4]
            train_ls.append(item)
            img = cv2.imread(item[0])
            h,w = img.shape[0], img.shape[1]
            M = cv2.getRotationMatrix2D((w/2,h/2), 30, 1)
            rotimg = cv2.warpAffine(img,M,(w,h))
            cv2.imwrite(filename+"1.jpg", rotimg) 
            train_ls.append([filename+"1.jpg",label])
            M = cv2.getRotationMatrix2D((w/2,h/2), -30, 1)
            rotimg = cv2.warpAffine(img,M,(w,h))
            cv2.imwrite(filename+"2.jpg", rotimg) 
            train_ls.append([filename+"2.jpg",label])
            crop_img = img[75:h-75, 75:w-75]
            cv2.imwrite(filename+"3.jpg", crop_img) 
            train_ls.append([filename+"3.jpg",label])
            flip_img = cv2.flip(img, flipCode=1)
            cv2.imwrite(filename+"4.jpg", flip_img) 
            train_ls.append([filename+"4.jpg",label])
            T = np.float32([[1, 0, w / 4], [0, 1, h / 4]]) 
            trans_img = cv2.warpAffine(img, T, (w, h)) 
            cv2.imwrite(filename+"5.jpg", trans_img) 
            train_ls.append([filename+"5.jpg",label])
            crop_img = img[50:h-50, 50:w-50]
            cv2.imwrite(filename+"6.jpg", crop_img) 
            train_ls.append([filename+"6.jpg",label])
            crop_img = img[100:h-100, 100:w-100]
            cv2.imwrite(filename+"7.jpg", crop_img) 
            train_ls.append([filename+"7.jpg",label])
        elif i<vl:
            valid_ls.append(item)
        else:
            test_ls.append(item)
        i+=1
    label+=1

print(len(train_ls),len(valid_ls),len(test_ls))

train_df = pd.DataFrame(train_ls, columns = ['path', 'label'])  
valid_df = pd.DataFrame(valid_ls, columns = ['path', 'label'])  
test_df = pd.DataFrame(test_ls, columns = ['path', 'label'])  
train_df.to_csv("train.csv")
valid_df.to_csv("valid.csv")
test_df.to_csv("test.csv")
