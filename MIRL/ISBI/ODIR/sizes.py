import os
import pandas as pd
import csv
from PIL import Image
from collections import Counter

# data = pd.read_csv('ODIR-5K_Training_Annotations.csv')
#print(data.head())
#print(data.tail())

path = 'reduced_size/'
print(path)
tmp = os.listdir(path)
print(len(tmp))
print(tmp[0])

a = []
names = []
sz = []
for i in range(len(tmp)):
#for i in range(5):
    jpgfile= Image.open(path + tmp[i])
    names.append(tmp[i])
    pos = jpgfile.size
    sz.append(pos)
    a.append(pos)
    #print(jpgfile.bits, jpgfile.size, jpgfile.format)

print("len(a)",len(a))
######################################################################################################################################
######################################################################################################################################
######################################################################################################################################
# print(Counter(a).keys()) 
#print(Counter(a).values())

size_keys= Counter(a).keys()
print("c type", type(size_keys))
size_keys_list = list(size_keys)
print("c type", size_keys_list[0])
print("len of lc", len(size_keys_list))


size_values = Counter(a).values()
print("c type", type(size_values))
size_values_list = list(size_values)
print("c type", size_values_list[0])
print("len of lc", len(size_values_list))
print("sum of list", sum(size_values_list))

######################################################################################################################################
######################################################################################################################################
######################################################################################################################################

sorted_keys_list = sorted(size_keys_list)
sorted_values_list = sorted(size_values_list)
print(sorted_keys_list, sorted_values_list)

rows = zip(sorted_keys_list,sorted_values_list,size_keys_list,size_values_list)

with open("sizes.csv", "w") as f:
    writer = csv.writer(f)
    for row in rows:
        writer.writerow(row)

rz = zip(names, sz)
with open("nm_sz.csv", "w") as f:
    writer = csv.writer(f)
    for row in rz:
        writer.writerow(row)