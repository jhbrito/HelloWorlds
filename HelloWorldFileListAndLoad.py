import os
import PIL.Image as PImage
import matplotlib.pyplot as plt


pasta="membrane/train/image"

lista_de_ficheiros = os.scandir(pasta)

for ficheiro in lista_de_ficheiros:
    ficheiro_path = os.path.join(pasta, ficheiro.name)
    image = PImage.open(ficheiro_path)
    plt.imshow(image)
print("End")


