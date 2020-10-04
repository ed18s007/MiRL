import numpy as np
from numpy.linalg import eig, svd

from matplotlib import pyplot
import os

def mat_trans(image):
    trans_mat = [[image[j][i] for j in range(len(image))] for i in range(len(image[0]))] 
    return np.array(trans_mat)

def mat_mult(mat_1, mat_2):
    m, n = mat_1.shape
    p, q = mat_2.shape
    result = np.zeros([m,q])
    if n != p:
        print("Check dimensions of matrix")
    result = [[sum(a * b for a, b in zip(A_row, B_col))  
                        for B_col in zip(*mat_2)] 
                                for A_row in mat_1] 
  
    return np.array(result)

def u_sigma_vt(u_vector, sigma, v_vector):
    mat_1 = np.expand_dims(u_vector,axis=0)
    mat_2 = np.expand_dims(v_vector,axis=0)
    u_sigma = np.transpose(mat_1)*sigma
    result = np.dot(u_sigma, mat_2)
    return result

def calc_u_sig_v(a,v,sigma):
    sigma_inv = np.zeros([64,64])
    sig_sqrt = np.sqrt(sigma)
    for i in range(64):
        sigma_inv[i][i] = 1.0/sig_sqrt[i]
    av = np.dot(a,v)
    u = np.dot(av,sigma_inv)
    return u 

def svd_img(path, precs = 0.0000001):
    image = pyplot.imread(path)
    image = np.array(image,dtype=np.float32)
    # image = image[:3,:3]
    # pyplot.imshow(image, pyplot.cm.gray)
    # pyplot.show()
    # image = image/255.0 
    # pyplot.imshow(image, pyplot.cm.gray)
    # pyplot.show()
    image_trans = np.transpose(image)
 
    img_img_t = np.dot(image, image_trans)
    img_t_img = np.dot(image_trans, image)

    values_1 , vectors_1 = eig(img_img_t)
    values_2 , vectors_2 = eig(img_t_img)
    print(image)
    print(values_1)
    print(values_2)
    vectors_1 = calc_u_sig_v(image, vectors_2, values_2)

    for i,lmbda in enumerate(values_2):
        if i==0:
            u_s_vt = u_sigma_vt(vectors_1[i],np.sqrt(values_2[i]), vectors_2[i])
            # pyplot.imshow(u_s_vt, pyplot.cm.gray)
            # pyplot.show()
        if (lmbda>precs) and i>0:
            # print(i,lmbda)
            u_s_vt += u_sigma_vt(vectors_1[i],np.sqrt(values_2[i]), vectors_2[i])
            # pyplot.imshow(u_s_vt, pyplot.cm.gray)
            # pyplot.show()
    return u_s_vt

# path 
path = "Dataset_Question1/1/1.pgm"
# fin_img = svd_img(path)
# pyplot.imshow(fin_img, pyplot.cm.gray)
# pyplot.show()

def u_sig_v(u,s,vt):
    sigma_mat = np.zeros([64,64])
    for i in range(64):
        sigma_mat[i][i] = s[i]
    av = np.dot(u,sigma_mat)
    fin = np.dot(av,vt)
    return fin 

def u_sig_v_vect(u_vector, sigma, v_vector):
    mat_1 = np.expand_dims(u_vector,axis=0)
    mat_2 = np.expand_dims(v_vector,axis=0)
    u_sigma = np.transpose(mat_1)*sigma
    result = np.dot(u_sigma, mat_2)
    return result

image = pyplot.imread(path)
image = image/255.0
pyplot.imshow(image, pyplot.cm.gray)
pyplot.show()

image = np.array(image,dtype=np.float32)

u, s, vt = np.linalg.svd(image, full_matrices=True)
print(s)
print(u.shape, s.shape, vt.shape)
fin_img = u_sig_v(u, s, vt )
# pyplot.imshow(fin_img, pyplot.cm.gray)
# pyplot.show()

u_h = np.squeeze(np.hsplit(u,64))
v_h = np.squeeze(np.hsplit(np.transpose(vt),64))

# first_vext_img = u_sig_v_vect(u_h[0], s[0], v_h[0])
# pyplot.imshow(first_vext_img, pyplot.cm.gray)
# pyplot.show()
top_eig_vals = 6

for i in range(top_eig_vals):
    if i==0:
        u_s_vt = u_sig_v_vect(u_h[i], s[i], v_h[i])
        pyplot.imshow(u_s_vt, pyplot.cm.gray)
        pyplot.show()
    elif i>0:
        print(i)
        u_s_vt += u_sig_v_vect(u_h[i], s[i], v_h[i])
        pyplot.imshow(u_s_vt, pyplot.cm.gray)
        pyplot.show()

# dir = os.listdir("Dataset_Question1")
# for fl in dir:
#     a = os.listdir("Dataset_Question1"+"/" + str(fl))
#     print(a)
#     for fls in a:
#         path = "Dataset_Question1"+"/" + str(fl) + "/" + fls
#         print(path)
#         fin_img = svd_img(path)
#         pyplot.imshow(fin_img, pyplot.cm.gray)
#         pyplot.show()