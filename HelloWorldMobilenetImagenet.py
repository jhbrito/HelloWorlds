#from __future__ import absolute_import, division, print_function, unicode_literals
#from keras.applications.mobilenet_v2 import MobileNetV2
#from keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
#from keras import utils as kutils
#from keras.preprocessing import image as kimage
import tensorflow as tf
import numpy as np
import PIL.Image as PImage
import matplotlib.pyplot as plt

model = tf.keras.applications.mobilenet_v2.MobileNetV2(weights='imagenet')
IMAGE_RES = 224

image_path  = tf.keras.utils.get_file('grace_hopper.jpg',  'https://storage.googleapis.com/download.tensorflow.org/example_images/grace_hopper.jpg')
labels_path = tf.keras.utils.get_file('ImageNetLabels.txt','https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
imagenet_labels = np.array(open(labels_path).read().splitlines())

image1 = PImage.open(image_path).resize((IMAGE_RES, IMAGE_RES))
plt.imshow(image1)
plt.axis('off')
_ = plt.title("Original image")
plt.show()
x1 = np.array(image1)
print(x1.shape)
x1 = x1[np.newaxis, ...]
print(x1.shape)
x1 = x1 / 255.0
result1 = model.predict(x1)
print(result1.shape)
predicted_class1 = np.argmax(result1[0], axis=-1)
print(predicted_class1)
predicted_class_name1 = imagenet_labels[predicted_class1+1]
print(predicted_class_name1)

image2 = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
x2 = tf.keras.preprocessing.image.img_to_array(image2)
x2 = np.expand_dims(x2, axis=0)
x2 = tf.keras.applications.mobilenet_v2.preprocess_input(x2)
result2 = model.predict(x2)
predicted_class_decoded2 = tf.keras.applications.mobilenet_v2.decode_predictions(result2, top=1)
predicted_class_name2 = predicted_class_decoded2[0][0][1]

predicted_class2 = np.argmax(result2[0], axis=-1)
print(predicted_class2)
results=np.concatenate((result1, result2), axis=0)

plt.imshow(image2)
plt.axis('off')
_ = plt.title("Prediction: " + predicted_class_name2.title())
plt.show()
