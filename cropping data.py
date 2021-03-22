from main import *
import os

#v1 = VideoStream(src=0).start()

images_path = r'/home/jap01/Desktop/TD_RGB_A_Set4'
path = r'/home/jap01/Desktop/copy_TD_RGB_A_Set4'
count =0
def addtodatabase(cropped_img,name,folder):
    folder_path = path + '/' + folder
    folders = os.listdir(path)
    if folder in folders:
        print("as")
    else:
        os.mkdir(folder_path)
    cv2.imwrite(folder_path + "/" + str(name) + ".jpg",cropped_img)

for i in os.listdir(images_path):
    folder = i
    if i[:-3] != "txt":
        for j in os.listdir(images_path + '/' + i):
            frame = cv2.imread(images_path  + '/' + i + '/' + j)
            frame = imutils.resize(frame, width=400)

            cv2.imshow("frame", frame)
            try:
                bndbox = detect_face(frame, facenet)
                # print(bndbox)
                index = 0
                for x1, y1, x2, y2 in bndbox:
                    if index == 1:
                        break
                    # (startx, starty, endx, endy) = box
                    #img = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cropped_img = frame[y1:y2, x1:x2]
                    cv2.imshow("face",cropped_img)
                    cv2.waitKey(1000)
                    count = count+1
                    addtodatabase(cropped_img,count,folder)
                    index = 1
            except:
                cv2.imshow("face", cropped_img)
                cv2.waitKey(1000)
                count = count + 1
                addtodatabase(frame, count, folder)
            print(count)
    #cv2.imshow("faces", frame)
    #key = cv2.waitKey(1) & 0xFF
    #if key == ord("q"):
    #    break

cv2.destroyAllWindows()