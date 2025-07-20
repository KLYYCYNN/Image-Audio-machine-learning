import numpy as np
from PIL import Image
import os

# %%

def img_ratio(img):
    return max( img.size[0]/img.size[1], img.size[1]/img.size[0] )
    

def crop_300n(img):
    img_res = np.array(img.size)
    argmin = np.argmin(img_res)
    s = img_res[argmin] // 300
    tb, lr = img_res[1] - s*300, img_res[0] - s*300
    b, r = tb//2, lr//2
    t, l = tb-b, lr-r
    b, r = img_res[1]-b, img_res[0]-r
    img = img.crop((l, t, r, b))
    return img


def fill_300n(img):
    img_arr = np.array(img)
    img_res = np.delete(np.array(img_arr.shape), -1)
    argmin = np.argmin(img_res)
    n = int(np.ceil( img_res[argmin]/300 ))
    n1 = (300*n - img_res[argmin]) // 2
    n2 = (300*n - img_res[argmin]) - n1
    k1, k2 = np.zeros((n1, img_res[1-argmin], 3)), np.zeros((n2, img_res[1-argmin], 3))
    img_arr = np.insert(img_arr, 0, k1, axis = argmin)
    img_arr = np.insert(img_arr, -1, k2, axis = argmin)
    return Image.fromarray( img_arr.astype(np.uint8) )


def resize_B3(img):
    if abs( img.size[0] - img.size[1] ) > 300:
        img = crop_300n( fill_300n(img) )
    else:
        img = crop_300n(fill_300n( fill_300n(img) ))
    return img.resize((300,300))


def training_data( raw_dir, trg_dir, prefix ): #convert pictures to 300 by 300
    file_names = os.listdir(raw_dir)
    nfile = len(file_names)
    zero, a, b, c, d = "0", "--", "  ", ">", "|"
    for i in range(nfile):
        img = Image.open( raw_dir + "/" + file_names[i] )
        file_name = prefix + int(2-np.floor(np.log10(i+1)))*zero + str(i+1) + ".jpg"
        resize_B3(img).save( trg_dir + "/" + file_name )
        progress = round((i+1)*20/nfile)
        print(f"{a*progress + c + b*(20-progress) + d + b*2}  Converting.....  {i+1}/{nfile}", end = '\r')
        

