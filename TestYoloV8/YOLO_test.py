from ultralytics import YOLO
import cv2

model = YOLO('yolov8n.pt')
result = model('../Images/parking1.png', show=True)
cv2.waitKey(0)

while True:
    success, img = cap.read()
    if not success:
        break
    # Doing detections using YOLOv8 frame by frame
    results = model(img, stream=True)
    # Once we have the results, loop through each bounding box
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            print(x1, y1, x2, y2)

            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            class_name = classNames[cls]

            list1 = []

            for index, row in px.iterrows():

                x1=int(row[0])
                y1=int(row[1])
                x2=int(row[2])
                y2=int(row[3])
                d=int(row[4])

            # Check if the detected object is a car
            if class_name.lower() == "car":
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
                label = f'{class_name}{conf}'
                t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
                c2 = x1 + t_size[0], y1 - t_size[1] - 3
                cv2.rectangle(img, (x1, y1), c2, [255, 0, 255], -1, cv2.LINE_AA)  # filled
                cv2.putText(img, label, (x1, y1 - 2), 0, 1, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)

    out.write(img)
    cv2.imshow("Image", img)
    cv2.setMouseCallback('Image', RGB)
    if cv2.waitKey(0) & 0xFF == ord('1'):
        break
out.release()