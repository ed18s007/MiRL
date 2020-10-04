import os
import pandas as pd
import csv
from PIL import Image
from collections import Counter

data = pd.read_csv('ODIR-5K_Training_Annotations.csv')
#print(data.head())
#print(data.tail())

path = 'ODIR-5K_Training_Dataset/'
print(path)
tmp = os.listdir(path)
print(len(tmp))
print(tmp[0])

size_set = {(250, 188), (320, 316), (727, 716), (741, 713), (758, 705), (763, 716),
            (764, 714), (770, 718), (800, 600), (868, 793), (924, 805), (925, 799),
            (929, 788), (930, 790), (947, 800), (949, 784), (951, 785), (955, 783),
        (957, 771), (959, 776), (960, 783), (972, 785), (1280, 960), (1320, 1065),
            (1375, 1085), (1380, 1382), (1440, 1080), (1444, 1444), (1467, 1471),
            (1468, 1300), (1468, 1470), (1468, 1472), (1469, 1470), (1470, 1137),
            (1470, 1471), (1470, 1472), (1470, 1473), (1471, 1473), (1476, 1483),
            (1536, 1152), (1600, 1400), (1620, 1444), (1624, 1232), (1677, 1260),
            (1725, 1721), (1747, 1312), (1892, 1422), (1895, 1424), (1920, 893),
            (1920, 894), (1920, 1088), (1920, 1296), (1936, 1296), (1956, 1934),
            (1974, 1483), (2048, 1536), (2057, 1545), (2065, 1850), (2090, 2080),
            (2100, 2100), (2124, 2056), (2139, 1607), (2142, 1609), (2144, 1424),
            (2196, 1958), (2196, 1960), (2232, 1677), (2272, 2048), (2286, 1769),
            (2304, 1728), (2304, 2048), (2373, 1837), (2400, 2400), (2414, 2416),
            (2460, 1904), (2464, 1632), (2480, 1919), (2560, 1920), (2584, 1951),
            (2584, 1990), (2584, 2000), (2592, 1728), (2592, 1944), (2736, 1824),
            (2785, 2350), (2940, 2920), (2960, 2935), (2976, 1984), (2976, 2976),
            (2992, 2000), (3216, 2136), (3264, 2448), (3280, 2480), (3456, 2304),
            (3456, 3456), (3504, 2336), (3696, 2448), (3888, 2592), (4288, 2848),
            (4496, 3000), (5184, 3456)}

a = []
#sorted_ss = sorted(size_set)
#print(sorted_ss)
list_ss = sorted(list(size_set))
#print(list_ss)
print(len(list_ss))
for i in range(len(tmp)):
#for i in range(5):
    jpgfile= Image.open(path + tmp[i])
    #sorted_ss.add(jpgfile.size)
    pos = list_ss.index(jpgfile.size)
    a.append(pos)
    #print(jpgfile.bits, jpgfile.size, jpgfile.format)
print(len(size_set))

#print(len(sorted_ss))
#print(a)

print(Counter(a).keys()) 
print(Counter(a).values())
#print(list_ss[69], list_ss[53], list_ss[93], list_ss[97], list_ss[81], list_ss[91], list_ss[87])
#print(list_ss[52], list_ss[55], list_ss[49], list_ss[27], list_ss[83], list_ss[48], list_ss[60])
#print(list_ss[82],list_ss[100],list_ss[88], list_ss[78], list_ss[51], list_ss[80], list_ss[42])
nn = [424, 390, 258, 124, 2146, 170, 212, 284, 524, 74, 267, 344, 20, 146, 102, 52, 124, 16, 78, 107,
      64, 30, 1, 1, 1, 1, 69, 198, 30, 20, 1, 1, 6, 52, 14, 38, 74, 56, 80, 10, 28, 6, 10, 1, 3, 8, 1,
      1, 1, 1, 32, 32, 16, 20, 1, 1, 68, 4, 14, 1, 1, 5, 12, 18, 2, 1, 4, 4, 1, 1, 20, 1, 1, 1, 1, 1, 10,
      2, 10, 4, 1, 1, 2, 5, 6, 2, 1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 4, 1, 1, 6, 1]
nns = sorted(nn,reverse=True)
print(nns)
print(list_ss[81], 2146)
print(list_ss[55], 524)
print(list_ss[69], 424)
print(list_ss[53], 390)
print(list_ss[83], 344)
print(list_ss[52], 284)
print(list_ss[27], 267)
print(list_ss[93], 258)
print(list_ss[87], 212)
print(list_ss[39], 198)
print(list_ss[91], 170)
print(list_ss[60], 146)
print(list_ss[97], 124)
print(list_ss[88], 124)
print(list_ss[80], 107)
print(list_ss[82], 102)

print(2146+524+424+390+344+284+267+258+212+198+170+146+
      124+124+107+102)
print(list_ss[89], 80)
print(list_ss[51], 78)
print(list_ss[25], 74)
print(list_ss[49], 74)
print(list_ss[86], 69)
print(list_ss[72], 68)
print(list_ss[42], 64)
print(80+78+74+74+69+68+64)
print(sum(nns))
print("5820+507 = ",5820+507)
print(list_ss[75], 56)
print(list_ss[100], 52)
print(list_ss[77], 52)
print(print("5820+507 + = ",5820+507+56+52+52))






