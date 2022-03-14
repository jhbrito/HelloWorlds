# example on how to visualize Sentinel 2 images
# you need to download the sentinel files before you run this script

import rasterio as rio
import matplotlib.pyplot as plt
import numpy as np

filePathB2 = 'S2A_MSIL2A_20171002T112111_N0205_R037_T29TNG_20171002T112230.SAFE\GRANULE\L2A_T29TNG_A011903_20171002T112230\IMG_DATA\R10m\L2A_T29TNG_20171002T112111_B02_10m.jp2'
filePathB3 = 'S2A_MSIL2A_20171002T112111_N0205_R037_T29TNG_20171002T112230.SAFE\GRANULE\L2A_T29TNG_A011903_20171002T112230\IMG_DATA\R10m\L2A_T29TNG_20171002T112111_B03_10m.jp2'
filePathB4 = 'S2A_MSIL2A_20171002T112111_N0205_R037_T29TNG_20171002T112230.SAFE\GRANULE\L2A_T29TNG_A011903_20171002T112230\IMG_DATA\R10m\L2A_T29TNG_20171002T112111_B04_10m.jp2'

# load blue channel
B2file = rio.open(filePathB2)
B2 = B2file.read(1)
# load green channel
B3file = rio.open(filePathB3)
B3 = B3file.read(1)
# load red channel
B4file = rio.open(filePathB4)
B4 = B4file.read(1)

# normalize channels to >=0.0 and <=1.0
# on each channel project mean - std to 0 and mean + std to 1
B2max= B2.max()
B2min = B2.min()
B2range = (B2max - B2min)/2
B2mean = B2.mean()
B2std = B2.std()
B2 = ((B2 - B2mean) / (1*B2std)) + .5
# B2 = (B2 - B2min) / B2range
B2[B2>1.0]=1
B2[B2<0.0]=0

B3max= B3.max()
B3min = B3.min()
B3range = (B3max - B3min)/2
B3mean = B3.mean()
B3std = B3.std()
B3 = ((B3 - B3mean) / (1*B3std)) + .5
# B3 = (B3 - B3min) / B3range
B3[B3>1.0]=1
B3[B3<0.0]=0

B4max= B4.max()
B4min = B4.min()
B4range = (B4max - B4min)/2
B4mean = B4.mean()
B4std = B4.std()
B4 = ((B4 - B4mean) / (1*B4std)) + .5
#B4 = (B4 - B4min) / B4range
B4[B4>1.0]=1
B4[B4<0.0]=0

# create RGB image as numpy array
(h, w) = B4.shape
S2Image = np.zeros((h, w, 3))
S2Image[:,:,0]=B4
S2Image[:,:,1]=B3
S2Image[:,:,2]=B2

# show RGB image
plt.imshow(S2Image)
plt.axis('off')
_ = plt.title("Minho")
plt.show()
