from main import *
import os

path = r'/home/jap01/PycharmProjects/face recogiition original/database'
def addtodatabase(cropped_img,name):
    #os.chdir(path)
    cv2.imwrite(path + "/" + name + ".jpg",cropped_img)

while True:
    frame = v1.read()
    frame = imutils.resize(frame, width=400)

    #cv2.imshow("frame", frame)
    bndbox = detect_face(frame, facenet)
    # print(bndbox)
    for x1, y1, x2, y2 in bndbox:
        # (startx, starty, endx, endy) = box
        img = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cropped_img = img[y1:y2, x1:x2]
        cv2.imshow("face",cropped_img)
        cv2.waitKey(1000)
        name = input("Enter Name : ")
        addtodatabase(cropped_img,name)
    break

    #cv2.imshow("faces", frame)
    #key = cv2.waitKey(1) & 0xFF
    #if key == ord("q"):
    #    break

cv2.destroyAllWindows()