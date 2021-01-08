import imutils
import numpy as np
import cv2
import glob
import os
import tensorflow
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from imutils.video import VideoStream

v1 = VideoStream(src=0).start()

path = r'/home/jap01/Desktop/Face-Mask-Detection/face_detector/deploy.prototxt'
weightpath = r'/home/jap01/Desktop/Face-Mask-Detection/face_detector/res10_300x300_ssd_iter_140000.caffemodel'
facenet  = cv2.dnn.readNet(path,weightpath)

def detect_face(frame,facenet):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224), (104.0, 177.0, 123.0))

    facenet.setInput(blob)
    detections = facenet.forward()
    #print(detections.shape)
    faces = []
    bndbox = []
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the detection
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        if confidence > 0.5:

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            face = frame[startY:endY, startX:endX]
            #face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            # print(face)
            #cv2.imshow("face",face)
            faces.append(face)
            bndbox.append((startX, startY, endX, endY))
            #print(bndbox)

    return bndbox





# while True:
#     frame = v1.read()
#     frame = imutils.resize(frame, width=400)
#
#     cv2.imshow("frame",frame)
#     bndbox = detect_face(frame,facenet)
#     #print(bndbox)
#     for x1,y1,x2,y2 in bndbox:
#         #(startx, starty, endx, endy) = box
#         cv2.rectangle(frame, (x1,y1), (x2, y2), (0,255,0) , 2)
#
#     cv2.imshow("faces",frame)
#     key = cv2.waitKey(1) & 0xFF
#     if key == ord("q") :
#         break
#
# cv2.destroyAllWindows()