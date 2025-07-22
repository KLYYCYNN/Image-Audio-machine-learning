import numpy as np
from PIL import Image
import os

# %%

def img_ratio(img):
    return max( img.size[0]/img.size[1], img.size[1]/img.size[0] )


def check_image_file(file_name):
    image_suffixes = [".jpg", "jpeg", ".JPG", "JPEG"]
    file_suffix = file_name[-4] + file_name[-3] + file_name[-2] + file_name[-1]
    for image_suffix in image_suffixes:
        if file_suffix == image_suffix:
                return False
    return True


def crop_300n(img):
    img_res = np.array(img.size)
    argmin = np.argmin(img_res)
    s = img_res[argmin] // 300
    tb, lr = img_res[1] - s*300, img_res[0] - s*300
    b, r = tb//2, lr//2
    t, l = tb-b, lr-r
    b, r = img_res[1]-b, img_res[0]-r
    return img.crop((l, t, r, b))


def fill_300n(img):
    img_arr = np.array(img)
    if img_arr.shape[2] != 3:
        return 0
    img_res = np.delete(np.array(img_arr.shape), -1)
    argmin = np.argmin(img_res)
    n = int(np.ceil( img_res[argmin]/300 ))
    n1 = (300*n - img_res[argmin]) // 2
    n2 = (300*n - img_res[argmin]) - n1
    k1, k2 = np.zeros((n1, img_res[1-argmin], img_arr.shape[2])), np.zeros((n2, img_res[1-argmin], img_arr.shape[2]))
    img_arr = np.insert(img_arr, 0, k1, axis = argmin)
    img_arr = np.insert(img_arr, -1, k2, axis = argmin)
    return Image.fromarray( img_arr.astype(np.uint8) )


def resize_B3(img):
    img = fill_300n(img)
    if img == 0:
        return img
    elif abs( img.size[0] - img.size[1] ) > 300:
        img = crop_300n(img)
    else:
        img = crop_300n(fill_300n( img ))
    return img.resize((300,300))


def training_data( src_dir, trg_dir, prefix, n_start=0 ): 
    file_names = os.listdir(src_dir)
    nfile = len(file_names)
    zero, a, b, c, d = "0", "--", "  ", ">", "|"
    for i in range(nfile):
        img = resize_B3( Image.open( src_dir + "/" + file_names[i] ) )
        if img != 0:
            file_name = prefix + int(2-np.floor(np.log10(i+1+n_start)))*zero + str(i+1+n_start) + ".jpg"
            img.save( trg_dir + "/" + file_name )
        else:
            os.remove(src_dir + "/" + file_names[i])
        progress = round((i+1)*20/nfile)
        print(f"{a*progress + c + b*(20-progress) + d}   Converting.....  {i+1}/{nfile}", end = '\r')
    print(f"{b*11}successfully added {len(os.listdir(trg_dir))} images{b*11}")
        
    
def clean_data_directory( directory ):
    file_names = os.listdir( directory )
    for i in range(len(file_names)):
        if check_image_file(file_names[i]):
            os.remove( directory + "/" + file_names[i] )
                   
        
#%%

#  crop_300n:  Crop the image to a square with side length being multiple of 300
#works the best this way to later resize it to 300*300.

#  fill_300n: fill the shorter side of the image with symmetric black strips to make its
#number of pixels multiple of 300, prevents too much of the longer side being cropped off.

#  training_data: takes in entire folder of pictures and change their dimension for neural
#network inputs, saving to another folder. src_dir and trg_dir are source and target directories.
#This function automatically name the processed pictures with a serial number, prefix + 000 
#by default, you can offset 000 to any number using the optional argument n_start.

# clean_data_directory: delete any file in a directory meant to store training data that is not a .jpg
#or a .jpeg file. Directories with other kinds of file will make training_data crush.
