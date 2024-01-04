import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
import time

model = YOLO("../YOLO-Weights/yolov8m.pt")
# fast = n, s, m, l, x = slow

def RGB (event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        colorsRGB = [x, y]
        print(colorsRGB)

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

cap = cv2.VideoCapture('../Videos/parking2.mp4')

my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height))

area1 = [(1130,693),(1177,783),(1321,774),(1248,688)]
area2 = [(1261,689),(1343,774),(1464,760),(1366,682)]
area3 = [(1379,681),(1476,759),(1590,745),(1485,671)]
area4 = [(1497,670),(1597,743),(1689,729),(1589,663)]
area5 = [(1615,661),(1701,725),(1780,711),(1680,654)]

while True:
    read, frame = cap.read()
    if not read:
        break

    frame = cv2.resize(frame, (1920, 1080))

    results = model.predict(frame)
    #   print(results)
    a = results[0].boxes.boxes
    px = pd.DataFrame(a).astype("float")
    #    print(px)

    list1 = []
    list2 = []
    list3 = []
    list4 = []
    list5 = []

    for index, row in px.iterrows():
        #        print(row)

        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])
        c = class_list[d]
        if 'car' in c:
            cx = int(x1 + x2) // 2
            cy = int(y1 + y2) // 2

            results1 = cv2.pointPolygonTest(np.array(area1, np.int32), ((cx, cy)), False)
            if results1 >= 0:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
                list1.append(c)

            results2 = cv2.pointPolygonTest(np.array(area2, np.int32), ((cx, cy)), False)
            if results2 >= 0:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
                list2.append(c)

            results3 = cv2.pointPolygonTest(np.array(area3, np.int32), ((cx, cy)), False)
            if results3 >= 0:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
                list3.append(c)

            results4 = cv2.pointPolygonTest(np.array(area4, np.int32), ((cx, cy)), False)
            if results4 >= 0:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
                list4.append(c)

            results5 = cv2.pointPolygonTest(np.array(area5, np.int32), ((cx, cy)), False)
            if results5 >= 0:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
                list5.append(c)

    a1 = (len(list1))
    a2 = (len(list2))
    a3 = (len(list3))
    a4 = (len(list4))
    a5 = (len(list5))

    lot = (a1 + a2 + a3 + a4 + a5)
    space = (5 - lot)
    print(space)

    if a1 == 1:
        cv2.polylines(frame, [np.array(area1, np.int32)], True, (0, 0, 255), 2)
        cv2.putText(frame, str('1'), (1177,783), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 1)
    else:
        cv2.polylines(frame, [np.array(area1, np.int32)], True, (0, 255, 0), 2)
        cv2.putText(frame, str('1'), (1177,783), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255), 1)

    if a2 == 1:
        cv2.polylines(frame, [np.array(area2, np.int32)], True, (0, 0, 255), 2)
        cv2.putText(frame, str('2'), (1343,774), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 1)
    else:
        cv2.polylines(frame, [np.array(area2, np.int32)], True, (0, 255, 0), 2)
        cv2.putText(frame, str('2'), (1343,774), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255), 1)

    if a3 == 1:
        cv2.polylines(frame, [np.array(area3, np.int32)], True, (0, 0, 255), 2)
        cv2.putText(frame, str('3'), (1476,759), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 1)
    else:
        cv2.polylines(frame, [np.array(area3, np.int32)], True, (0, 255, 0), 2)
        cv2.putText(frame, str('3'), (1476,759), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255), 1)

    if a4 == 1:
        cv2.polylines(frame, [np.array(area4, np.int32)], True, (0, 0, 255), 2)
        cv2.putText(frame, str('4'), (1597,743), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 1)
    else:
        cv2.polylines(frame, [np.array(area4, np.int32)], True, (0, 255, 0), 2)
        cv2.putText(frame, str('4'), (1597,743), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255), 1)

    if a5 == 1:
        cv2.polylines(frame, [np.array(area5, np.int32)], True, (0, 0, 255), 2)
        cv2.putText(frame, str('5'), (1701,725), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 1)
    else:
        cv2.polylines(frame, [np.array(area5, np.int32)], True, (0, 255, 0), 2)
        cv2.putText(frame, str('5'), (1701,725), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255), 1)

    cv2.putText(frame, str(space), (23, 30), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 2)

    cv2.imshow("RGB", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()
# stream.stop()