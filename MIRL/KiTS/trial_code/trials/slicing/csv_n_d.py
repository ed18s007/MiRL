import os
import random
import numpy as np 
from tqdm import tqdm
from tqdm import tqdm_notebook, tnrange
import pandas as pd 
# train_data_folder_path = "datan/train_slices/X/"
# train_labels_folder_path = "datan/train_slices/y/"
# test_data_folder_path = "datan/test_slices/X/"
# test_labels_folder_path = 'datan/test_slices/y/'
# train_data_folder_path = "datan/train_3d/X/"
# train_labels_folder_path = "datan/train_3d/y/"
# test_data_folder_path = "datan/test_3d/X/"
# test_labels_folder_path = 'datan/test_3d/y/'
# aug_kid_x = 'augk2/X/'
# aug_kid_y = 'augk2/y/'
# aug_tum_x = 'augt/X/'
# aug_tum_y = 'augt/y/'
pred = 'datan/pred_slices/X/'
import csv
def csvwriter(csv_name,path):
    csvfile = csv_name
    tmp = os.listdir(path)
    full_list = [os.path.join(path,i) for i in tmp]
    time_sorted_list = sorted(full_list, key=os.path.getmtime)

    print(time_sorted_list)

    #Assuming res is a flat list
    with open(csvfile, "w") as output:
        writer = csv.writer(output, lineterminator='\n')
        for val in time_sorted_list:
            data = str(val)
            writer.writerow([data]) 

tmp = os.listdir(pred)
ls = []
print("Number of files : " + str(len(tmp)))
for j in range(0, len(tmp)):
    path = os.path.join(pred+ tmp[j])
    ls.append(path)
time_sorted_list = sorted(ls, key=os.path.getmtime)
# csvwriter("augk2_x.csv", aug_kid_x)
# csvwriter("augk2_y.csv", aug_kid_y)
# csvwriter("augt_x.csv", aug_tum_x)
# csvwriter("augt_y.csv", aug_tum_y)

# df = pd.DataFrame(list(zip(lkmin, lkmax, ltmin, ltmax)), 
#                columns =['lkmin', 'lkmax','ltmin', 'ltmax'])


# df.to_csv('new.csv')

df = pd.DataFrame(time_sorted_list)
df.to_csv('pred.csv')

# augx = pd.read_csv('augk2_x.csv')
# augx = augx.values.tolist()
# sx = sorted(augx)
# # print(sx)

# df = pd.DataFrame(sx)
# df.to_csv('sx.csv')

# augy = pd.read_csv('augk2_y.csv')
# augy = augy.values.tolist()
# sy = sorted(augy)
# # print(sx)

# df = pd.DataFrame(sy)
# df.to_csv('sy.csv')