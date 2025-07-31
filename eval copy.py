#%%
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model 
from PIL.Image import open
plt.rcParams['figure.figsize'] = (8, 6)
#%%
model = load_model( "models/vehicle_recognition_algorithm1_2.keras" )
#%%

def get_serial(n):
    return int(2-np.floor(np.log10(n)))*"0" + str(n)


def eval_single_cat(model, folder, classes, correct_class, save=False, exp_path=None, n_start=None, plot=True):
    n_fail = 0
    classes.sort()
    correct_index = classes.index(correct_class)
    image_list = os.listdir(folder)
    image_paths = [ os.path.join(folder, img) for img in image_list ]
    test_data = np.array( [ np.array(open(i)) for i in image_paths ] )
    prd_arr = model.predict(test_data)
    predictions = np.array([ np.argmax(i) for i in prd_arr ])
    for i in range(len(predictions)):
        if predictions[i] != correct_index:
            n_fail += 1
            fig, (ax1, ax2) = plt.subplots(1, 2)
            ax1.imshow(test_data[i])
            ax2.bar(classes, prd_arr[i])
            ax1.set_axis_off()
            ax1.set_title(f"{classes[predictions[i]]} ({correct_class})")
            if save:
                plt.savefig( exp_path + "/" + "FailCase" + get_serial(n_start+n_fail) + ".jpg" )
            if plot:
                plt.show()
            else:
                plt.close()
    print(f"class {chr(39)}{correct_class}{chr(39)}  accuracy: {np.round(1-n_fail/len(image_list),4)}")
    return n_fail


def eval_ds(model, ds, save=False, exp_path=None, plot=True):
    if save:
        assert exp_path != None, "Where do you want to save the cases of failure?"
    classes = os.listdir(ds)
    classes.sort()
    n_start, n_total = 0, 0
    for cls in classes:
        folder = os.path.join(ds, cls)
        eval_args = {
            "model": model,
            "folder": folder,
            "classes": classes,
            "correct_class": cls,
            "plot": plot,
        }
        if save:
            eval_args["save"], eval_args["exp_path"], eval_args["n_start"] = True, exp_path, n_start
        n_fail_i = eval_single_cat(**eval_args)
        n_start += n_fail_i
        n_total += len(os.listdir(folder))
    print(f"overall accuracy: {np.round(1-n_start/n_total, 4)}")

eval_ds(model, "data/Testing", save=True, exp_path="failure cases", plot = False)

# %%
def predict_image( model, image_path, classes ):
    classes.sort()
    image = np.array( open(image_path) )
    prediction = model.predict( np.array([image]) ).reshape(-1)
    assert len(prediction) == len(classes), "Wrong number of categories."
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(image)
    ax2.bar(classes, prediction)
    ax1.set_axis_off()
    ax1.set_title(f"{classes[np.argmax(prediction)]}", fontsize = 15)
    plt.show()


def predict_folder( model, folder_path, classes ):
    classes.sort()
    image_list = os.listdir(folder_path)
    image_paths = [os.path.join(folder_path, img) for img in image_list]
    data_arr = np.array([ np.array(open(image)) for image in image_paths ])
    prd_arr = model.predict(data_arr)
    assert len(prd_arr[0]) == len(classes), "Wrong number of categories."
    predictions = np.array([ np.argmax(i) for i in prd_arr ])
    for i in range(len(predictions)):
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(data_arr[i])
        ax2.bar(classes, prd_arr[i])
        ax1.set_axis_off()
        ax1.set_title(f"{classes[predictions[i]]}", fontsize = 15)
        plt.show()

predict_folder(model, "data/Testing/van", ["SUV", "van", "pickup"])
#%%
print(os.listdir("data/Testing"))
# classes = os.listdir("data/Testing")
# classes.sort()
# image_list = os.listdir(test_folder)
# image_paths = [ os.path.join( test_folder, img ) for img in image_list ]
# test_data = np.array([ np.array( open(i) ) for i in image_paths ])
# correct_class = 0
# prd_arr = model.predict(test_data)
# predictions = np.array([ np.argmax(i) for i in prd_arr ])
# for i in range(len(predictions)):
#     if predictions[i] != correct_class:
#         fig, (ax1, ax2) = plt.subplots(1, 2)
#         ax1.imshow(test_data[i])
#         ax2.bar(classes, prd_arr[i])
#         ax1.set_axis_off()
#         ax1.set_title(f"{classes[predictions[i]]} ({classes[correct_class]})")
#         plt.show()

# %%
