import functions
import matplotlib.pyplot as plt
import numpy as np

img_path = "/Users/mac/Deep_Learning/SinGAN/conSinGAN/ConSinGAN/Images/Generation/Batik_156.jpg"
opt = type("opt", (object,), {"nc_im" : 3, "not_cuda" : True})

batik_188 = functions.read_image_dir(img_path, opt)
batik_188 = functions.convert_image_np(batik_188)*255.0
print(batik_188, batik_188.shape)
plt.imshow(functions.shuffle_grid(batik_188).astype(np.uint8))
plt.show()
