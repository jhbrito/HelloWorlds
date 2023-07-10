import yolov5  # pip install yolov5
import os
import cv2
import numpy as np
import argparse
import time


parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, help='Path to a video or a sequence of image.',
                    default=os.path.join('Files', 'vtest.avi'))
args = parser.parse_args()

# load model
model = yolov5.load('../yolov5n.pt')
print(model.names)
model.conf = 0.33

cap = cv2.VideoCapture(args.input)
i = 0
begin_time = -1

while True:
    last_begin_time = begin_time
    begin_time = begin_read_time = time.time_ns()
    ret, image = cap.read()
    if not ret:
        break

    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = model(imageRGB)
    output = image.copy()
    cv2.putText(img=output,
                text=str(i),
                org=(0, 30),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1.0,
                color=(0, 255, 0),
                thickness=2)
    last_frame_time = (begin_time - last_begin_time) / 1000000000.0
    cv2.putText(img=output,
                text='{:.2f} f/s'.format(1 / last_frame_time, ),
                org=(output.shape[1] - 150, 30),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1.0,
                color=(0, 255, 0),
                thickness=2)
    for pred in enumerate(results.pred):
        im = pred[0]
        im_boxes = pred[1]
        for *box, conf, cls in im_boxes:
            box_class = int(cls)
            conf = float(conf)
            # frameID, trackID, x, y, w, h, score,-1,-1,-1
            x = float(box[0])
            y = float(box[1])
            w = float(box[2]) - x
            h = float(box[3]) - y
            pt1 = np.array(np.round((float(box[0]), float(box[1]))), dtype=int)
            pt2 = np.array(np.round((float(box[2]), float(box[3]))), dtype=int)
            box_color = (255, 0, 0)
            cv2.rectangle(img=output,
                          pt1=pt1,
                          pt2=pt2,
                          color=box_color,
                          thickness=1)
            text = "{}:{:.2f}".format(results.names[box_class], conf)
            cv2.putText(img=output,
                        text=text,
                        org=np.array(np.round((float(box[0]), float(box[1]-1))), dtype=int),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5,
                        color=box_color,
                        thickness=1)

    cv2.imshow("YOLOv5", output)
    i = i + 1

    # Exit if ESC pressed
    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break


