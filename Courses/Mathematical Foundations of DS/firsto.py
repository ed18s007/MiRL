import numpy as np
from numpy.linalg import eig, svd
from numpy.linalg import norm as linorm

from matplotlib import pyplot
import os

# from PIL import Image

"""
INPUT: Single Eigen vector of U, 
    corresponding eigen value and eigenvector of V
Output : Returns u*sigma*vt of same shape as original matrix(64,64)
"""
def u_sig_v_vect(u_vector, sigma, v_vector):
    mat_1 = np.expand_dims(u_vector,axis=0)
    mat_2 = np.expand_dims(v_vector,axis=0)
    u_sigma = np.transpose(mat_1)*sigma
    result = np.dot(u_sigma, mat_2)
    return result

"""
INPUT: image input path and number of top eigen values 
Output : Returns svd on image at path
"""
def svd_on_top_eig(path, top_eig_vals = 2):
    image = pyplot.imread(path)
    # pyplot.imshow(image, pyplot.cm.gray)
    # pyplot.show()

    image = np.array(image,dtype=np.float32)
    image = image/255.0
    u, s, vt = np.linalg.svd(image, full_matrices=True)

    u_h = np.squeeze(np.hsplit(u,64))
    v_h = np.squeeze(np.hsplit(np.transpose(vt),64))

    for i in range(top_eig_vals):
        if i==0:
            u_s_vt = u_sig_v_vect(u_h[i], s[i], v_h[i])
            # pyplot.imshow(u_s_vt, pyplot.cm.gray)
            # pyplot.show()
        elif i>0:
            u_s_vt += u_sig_v_vect(u_h[i], s[i], v_h[i])
            # pyplot.imshow(u_s_vt, pyplot.cm.gray)
            # pyplot.show()
    return u_s_vt

"""
INPUT: image, list of path of all representative images and
         list of class they represent
Output : predicted class by comparing norm 
"""
def find_img_cls(image, rpr_img_ls, rpr_cls):
    norm_min = np.inf
    for i, rpr_path in enumerate(rpr_img_ls):
        rpr_img = np.load(rpr_path + ".npy")
        res = image - rpr_img
        norm = np.sum(res**2)
        # norm = linorm(res, ord=2)

        if norm < norm_min:
            norm_min = norm 
            pred_cls = rpr_cls[i]
            # print(rpr_path, pred_cls, norm)
    return pred_cls
"""
INPUT: list of dataset images, corresponding classes,
    path of all representative images and list of class they represent
Output : Accuracy
"""
def check_accuracy(image_ls, class_ls, rpr_img_ls, rpr_cls):
    count = 0
    for i, path in enumerate(image_ls):  
        image = pyplot.imread(path) 
        image = np.array(image,dtype=np.float32)
        image = image/255.0
        image_cls_pred = find_img_cls(image, rpr_img_ls, rpr_cls)
        if int(image_cls_pred) == int(class_ls[i]) :
            count += 1
        else:
            print("misclassified image path and predicted ", path, image_cls_pred)
    print("count",count)
    return count/1.5

# Create empty lists to store
image_ls, class_ls, repr_img_ls, repr_cls_ls  = [], [], [], []
# Top eigen values to decompose
TOP_EIG_VAL = 64
dir = os.listdir("Dataset_Question1")
# Create directories to save files
try:
    os.mkdir("np_files")
    os.mkdir("representative_img")
except OSError:
    print ("Directories already present")
else:
    print ("Successfully created the directory ")

for fl in dir:
    a = os.listdir("Dataset_Question1"+"/" + str(fl))
    final_repr_image = np.zeros([64,64])
    for fls in a:
        path = "Dataset_Question1"+"/" + str(fl) + "/" + fls
        image_ls.append(path)
        class_ls.append(fl)
        # print(path)
        final_repr_image += svd_on_top_eig(path, TOP_EIG_VAL)
    final_repr_image = final_repr_image/10.0
    np_files_path = "np_files"+"/" + str(fl) #+ ".png"
    np.save(np_files_path, final_repr_image) 
    repr_img_ls.append(np_files_path)
    repr_cls_ls.append(str(fl))

    save_path = "representative_img"+"/" + str(fl) + ".png"
    pyplot.imsave(save_path, final_repr_image, cmap=pyplot.cm.gray)

    # pyplot.imshow(final_repr_image, pyplot.cm.gray)
    # pyplot.show()
# print(image_ls, class_ls, repr_img_ls, repr_cls_ls)
ch = check_accuracy(image_ls, class_ls, repr_img_ls, repr_cls_ls)
print("Accuracy percentage is : ",ch)