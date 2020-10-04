# importing pandas as pd 
import pandas as pd 
  
# Let's create the dataframe 
df = pd.DataFrame({'Date':['10/2/2011', '12/2/2011', '13/2/2011', '14/2/2011'], 
                    'Event':['Music', 'Poetry', 'Theatre', 'Comedy'], 
                    'Cost':[10000, 5000, 15000, 2000]}) 
  
# Let's visualize the dataframe 
print(df) 
# Function to insert row in the dataframe 
def Insert_row_(row_number, df, row_value): 
    # Slice the upper half of the dataframe 
    df1 = df[0:row_number] 
    # Store the result of lower half of the dataframe 
    df2 = df[row_number:] 
    # Inser the row in the upper half dataframe 
    df1.loc[row_number]=row_value 
    # Concat the two dataframes 
    df_result = pd.concat([df1, df2]) 
    # Reassign the index labels 
    df_result.index = [*range(df_result.shape[0])] 
    # Return the updated dataframe 
    return df_result
i=1
dfcopy = df.iloc[i:i+1,:].copy()
print("dfcopy",dfcopy)
dfcopy.iloc[:,1:3] = ['Mara',12345]
print("dfcopy",dfcopy)
newdf = pd.DataFrame(df.iloc[:0,:])
print("newdf", newdf)
newdf = pd.concat([newdf,dfcopy])
print("newdf", newdf)
newdf = pd.concat([newdf,dfcopy])
print("newdf", newdf)
newdf = pd.concat([newdf,dfcopy])
print("newdf", newdf)

from PIL import Image
import numpy as np 
import os
import pandas as pd 
import matplotlib.pyplot as plt 

def gen_patch_img(path, img_arr):
	height, width, ch = img_arr.shape 
	patch1 = patch2 = patch3 = patch4 = np.zeros([1632, 1632, 3], dtype = np.uint8)
	patch1 = img_arr[:1632,:1632,:]
	patch2 = img_arr[:1632,width-1632:,:]
	patch3 = img_arr[height-1632:,:1632,:]
	patch4 = img_arr[height-1632:,width-1632:,:]
	im1 = Image.fromarray(patch1)
	im1.save(path + 'p1.jpg')
	im2 = Image.fromarray(patch2)
	im2.save(path + 'p2.jpg')
	im3 = Image.fromarray(patch3)
	im3.save(path + 'p3.jpg')
	im4 = Image.fromarray(patch4)
	im4.save(path + 'p4.jpg')
	return patch1, patch2, patch3, patch4

filename = 'hypertension.csv'
data = pd.read_csv(filename)
print(data.head())

path = 'reduced_size/'
save_path = 'data_by_dis/hypertension/'
hype_data = data[['Left-Fundus', 'Right-Fundus']]

i = 0

dfcopy = data.copy()
df_fin = pd.DataFrame(dfcopy.iloc[:0,:])

left = []
right = []
for a, b in hype_data.itertuples(index=False):
    left.append(a)
    right.append(b)	
   # print(a, b) 
for l,r in zip(left,right):
	jpgfile = Image.open(path + l)
	pix = np.array(jpgfile)
	if((pix.shape[0] > 1632) or (pix.shape[1] > 1632)):
		gen_patch_img(save_path + l[:-4], pix)
		img = jpgfile.resize((1632,1632), Image.ANTIALIAS)
		img.save(save_path + l[:-4] + 'p5.jpg')
		for k in range(5):
			dfcopy.iloc[i:i+1,4:6] = [l + 'p' + str(k) +'.jpg', r + 'p' + str(k) +'.jpg']
			df_fin = pd.concat([df_fin,dfcopy])
		i += 1


	jpgfile = Image.open(path + r)
	pix = np.array(jpgfile)
	if((pix.shape[0] > 1632) or (pix.shape[1] > 1632)):
		gen_patch_img(save_path + r[:-4], pix)
		img = jpgfile.resize((1632,1632), Image.ANTIALIAS)
		img.save(save_path + r[:-4] + 'p5.jpg')


df_fin.to_csv('out.csv')
