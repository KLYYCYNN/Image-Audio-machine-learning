import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as img
# %%

basilica = img.open('img/basilica.jpg')
img1 = np.array(basilica)

#%%

fig, ax = plt.subplots()
ax.imshow(img1)
ax.set_axis_off()

plt.show()

#%%

gs_img1 = np.zeros((len(img1), len(img1[0])))
normalization = 255*np.sqrt(3)

for i in range(len(img1)):
    for j in range(len(img1[0])):
        gs_img1[i][j] = np.linalg.norm(img1[i][j])/normalization
        
# %%

fig, ax = plt.subplots()
ax.imshow(gs_img1, cmap = 'grey')
ax.set_axis_off()

plt.show()

int_gs_img1 = [[int(item) for item in row] for row in gs_img1*255]
int_gs_img1 = np.array(int_gs_img1).astype(np.uint8)

basilica_gs = img.fromarray(int_gs_img1, mode = 'L')
basilica_gs.save('img/basilica_gs.jpg')
