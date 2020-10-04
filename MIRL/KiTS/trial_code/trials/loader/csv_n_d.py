import os
import random
import numpy as np 
from tqdm import tqdm
from tqdm import tqdm_notebook, tnrange

train_data_folder_path = "datan/tr1_3d/X/"
train_labels_folder_path = "datan/tr1_3d/y/"
test_data_folder_path = "datan/tr2_3d/X/"
test_labels_folder_path = 'datan/tr2_3d/y/'

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

csvwriter("tr1_3d_X.csv", train_data_folder_path)
csvwriter("tr1_3d_y.csv", train_labels_folder_path)
csvwriter("tr2_3d_X.csv", test_data_folder_path)
csvwriter("tr2_3d_y.csv", test_labels_folder_path)

