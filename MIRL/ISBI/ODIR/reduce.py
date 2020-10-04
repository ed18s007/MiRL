from PIL import Image
import numpy as np 
import os
import csv

name = "0_left.jpg"
def my_indx(arr, width):
	first = 0
	for i in range(width):
		if(arr[i] == 0):
			first = i
		elif(arr[i]>0):
			break
	return first

def my_indy(arr, height):
	first = 0
	for i in range(height):
		if(arr[i] == 0):
			first = i
		elif(arr[i]>0):
			break
	return first

def get_coordinates(jpg_filename):
	jpgfile = Image.open(jpg_filename)
	# print(jpgfile.bits, jpgfile.size, jpgfile.format)
	width, height = jpgfile.size
	# print("width, height ",width, height )
	hfht, hfwd= int(height/2), int(width/2)
	# print(hfht, hfwd)
	pix = np.array(jpgfile)
	# print("pix shape",pix.shape)

	a0 = pix[hfht,:,0]
	ar0 = a0[::-1]

	a1 = pix[hfht,:,1]
	ar1 = a1[::-1]

	a2 = pix[hfht,:,2]
	ar2 = a2[::-1]

	x01 = my_indx(a0, width)
	x02 = width - my_indx(ar0, width)
	# print("x01, x02",x01, x02)

	x11 = my_indx(a1, width)
	x12 = width - my_indx(ar1, width)

	x21 = my_indx(a2, width)
	x22 = width - my_indx(ar2, width)

	x_left  = min(x01, x11, x21)
	x_right = min(x02, x12, x22)

	b0 = pix[:,hfwd,0]
	br0 = b0[::-1]

	b1 = pix[:,hfwd,1]
	br1 = b1[::-1]

	b2 = pix[:,hfwd,2]
	br2 = b2[::-1]

	y01 = my_indy(b0, height)
	y02 = height - my_indy(br0, height)

	y11 = my_indy(b1, height)
	y12 = height - my_indy(br1, height)

	y21 = my_indy(b2, height)
	y22 = height - my_indy(br2, height)

	y_down  = min(y01, y11, y21)
	y_up = min(y02, y12, y22)

	return (x_left, x_right, y_down, y_up, pix)

def new_image(name):
	x_left, x_right, y_down, y_up, pix = get_coordinates(name)
	r_img = pix[y_down :y_up , x_left :x_right,:]
	return r_img

path = 'ODIR-5K_Training_Dataset/'
tmp = os.listdir(path)
print(len(tmp))
# name = path + '4515_left.jpg'
# red_img = new_image(name)
# new_im = Image.fromarray(red_img)
# print(".........")
# new_im.save('reduced_size/' + '4515_left.jpg')

# for i in range(5):
for i in range(len(tmp)):
	name = path + tmp[i]
	red_img = new_image(name)
	new_im = Image.fromarray(red_img)
	new_im.save('reduced_size/' + tmp[i])

# with open('some.csv', 'w') as f:
#     writer = csv.writer(f)
#     writer.writerows(zip(bins, frequencies))







