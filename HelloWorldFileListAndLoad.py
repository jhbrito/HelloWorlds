import os
import PIL.Image as PImage
import matplotlib.pyplot as plt

pasta = "membrane/train/image"

list_of_files = os.scandir(pasta)

for file in list_of_files:
    file_path = os.path.join(pasta, file.name)
    image = PImage.open(file_path)
    plt.imshow(image, cmap=plt.cm.gray)
    plt.show()
print("End")
