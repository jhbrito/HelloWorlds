# Demo with a few examples of using OpenCV functions and UI
# packages: opencv-python
# uses lena: https://upload.wikimedia.org/wikipedia/en/7/7d/Lenna_%28test_image%29.png

import numpy as np
import cv2

print("Hello World OpenCV")
print("OpenCV Version:", cv2.__version__)

image = np.ones((256, 256), dtype="uint8")
image = image * 127
image[0:128, 0:128] = 0
image[128:, 128:] = 255
cv2.imshow("Image", image)
cv2.waitKey(0)

# Opening and Viewing an Image
import os.path

if os.path.isfile('lena.png'):
    print("Test Image File exist")
else:
    print("Test Image File does not exist; downloading...")
    import urllib.request as urllib_request

    urllib_request.urlretrieve("https://upload.wikimedia.org/wikipedia/en/7/7d/Lenna_%28test_image%29.png", "lena.png")

image = cv2.imread("./lena.png")
cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
cv2.imshow("Image RGB", rgb_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


def viewImage(image, name_of_window):
    cv2.namedWindow(name_of_window, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(name_of_window, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


viewImage(image, "Lena")

# Edit pixels
edited = image.copy()
edited[200:390, 200:360, 0] = 255
viewImage(edited, "Lena edited")

# Cropping
cropped = image[200:390, 200:360]
viewImage(cropped, "Lena cropped")

# Resizing
scale_percent = 10  # percent of original size
width = int(image.shape[1] * scale_percent / 100)
height = int(image.shape[0] * scale_percent / 100)
dim = (width, height)
resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
viewImage(resized, "Lena resized to {}%".format(scale_percent))

# Drawing a Rectangle
output = image.copy()
cv2.rectangle(output, (200, 200), (360, 390), (255, 0, 0), 10)
viewImage(output, "Lena with a rectangle")

# Drawing a line
cv2.line(output, (256, 390), (256, 512), (0, 0, 255), 5)
viewImage(output, "Lena with a line")

# Writing on an image
cv2.putText(output, "Lena", (360, 390), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
viewImage(output, "Lena with text")

# Saving an image
cv2.imwrite("./output.jpg", output)

# Blurring/Smoothing
blurred = cv2.GaussianBlur(image, (15, 15), 0)
viewImage(blurred, "Lena blurred")

# Rotating
(h, w, d) = image.shape
center = (w // 2, h // 2)
rot = 45
M = cv2.getRotationMatrix2D(center, rot, 1.0)
rotated = cv2.warpAffine(image, M, (w, h))
viewImage(rotated, "Lena rotated by {} degrees".format(rot))

# Blend
alpha_slider_max = 100


def on_trackbar_weight(val):
    alpha = val / alpha_slider_max
    beta = (1.0 - alpha)
    blend = cv2.addWeighted(image, alpha, rotated, beta, 0.0)
    cv2.imshow('Lena blended', blend)


cv2.namedWindow('Lena blended')
trackbar_name = 'Alpha 0 - {}'.format(alpha_slider_max)
cv2.createTrackbar(trackbar_name, 'Lena blended', 50, alpha_slider_max, on_trackbar_weight)
on_trackbar_weight(50)
cv2.waitKey()
cv2.destroyWindow('Lena blended')

# Grayscaling
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
viewImage(gray_image, "Lena gray-scale")

# Thresholding
threshold_slider_max = 255
threshold = 200
ret, threshold_image = cv2.threshold(gray_image, threshold, 255, 0)


def on_trackbar_threshold(val):
    threshold = val
    ret, threshold_image = cv2.threshold(gray_image, threshold, 255, 0)
    cv2.imshow("Lena thresholded", threshold_image)


cv2.namedWindow("Lena thresholded")
trackbar_name = "Threshold 0 - {}".format(threshold_slider_max)
cv2.createTrackbar(trackbar_name, "Lena thresholded", threshold, threshold_slider_max, on_trackbar_threshold)
on_trackbar_threshold(threshold)
cv2.waitKey()
cv2.destroyWindow("Lena thresholded")

# Contours
contours, hierarchy = cv2.findContours(threshold_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
image_with_contours = image.copy()
cv2.drawContours(image_with_contours, contours, -1, (255, 0, 0), 1)
viewImage(image_with_contours, "Lena contours")

# Face Detection
face_cascade = cv2.CascadeClassifier('venv\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml')
faces = face_cascade.detectMultiScale(gray_image)
print("Lena with {} faces detected".format(len(faces)))
image_faces = image.copy()
for (x, y, w, h) in faces:
    cv2.rectangle(image_faces, (x, y), (x + w, y + h), (0, 255, 0), 2)
viewImage(image_faces, "Lena with {} faces detected".format(len(faces)))
