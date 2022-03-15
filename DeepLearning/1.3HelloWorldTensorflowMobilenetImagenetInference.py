import tensorflow as tf
import tensorflow.keras as tf_keras
import numpy as np
import PIL.Image as PImage
import matplotlib.pyplot as plt

model = tf_keras.applications.mobilenet_v2.MobileNetV2(weights='imagenet')
model.summary()

IMAGE_RES = 224

labels_path = tf_keras.utils.get_file('ImageNetLabels.txt','https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
imagenet_labels = np.array(open(labels_path).read().splitlines())

image_path = tf_keras.utils.get_file('grace_hopper.jpg',  'https://storage.googleapis.com/download.tensorflow.org/example_images/grace_hopper.jpg')
image1 = PImage.open(image_path).resize((IMAGE_RES, IMAGE_RES))
image_path = tf_keras.utils.get_file('STOP.jpg',  'https://upload.wikimedia.org/wikipedia/commons/f/f9/STOP_sign.jpg')
image2 = PImage.open(image_path).resize((IMAGE_RES, IMAGE_RES))

plt.imshow(image1)
plt.axis('off')
_ = plt.title("Original image")
plt.show()
x1 = np.array(image1)
x2 = np.array(image2)

print("x1.shape:",x1.shape)
x=np.zeros( (2,IMAGE_RES,IMAGE_RES,3))
#x = x1[np.newaxis, ...]
x[0] = x1
x[1] = x2

print("batch shape", x.shape)
x = x / 255.0
x = x * 2.0
x = x - 1.0

result = model.predict(x)
print(result.shape)
predicted_class1 = np.argmax(result[0], axis=-1)
print("predicted_class1:", predicted_class1)
predicted_class2 = np.argmax(result[1], axis=-1)
print("predicted_class2:", predicted_class2)

predicted_class_name1 = imagenet_labels[predicted_class1+1]
print("predicted_class_name1:", predicted_class_name1)
predicted_class_name2= imagenet_labels[predicted_class2+1]
print("predicted_class_name2:", predicted_class_name2)

image3 = tf_keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
x3 = tf_keras.preprocessing.image.img_to_array(image3)
x3 = np.expand_dims(x3, axis=0)
x3 = tf_keras.applications.mobilenet_v2.preprocess_input(x3)
result3 = model.predict(x3)
predicted_class_decoded3 = tf_keras.applications.mobilenet_v2.decode_predictions(result3, top=1)
predicted_class_name3 = predicted_class_decoded3[0][0][1]

predicted_class3 = np.argmax(result3[0], axis=-1)
print("Class:", predicted_class3)

plt.imshow(image3)
plt.axis('off')
_ = plt.title("Prediction: " + predicted_class_name3.title())
plt.show()
