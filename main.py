import time
import torch
import cv2
import serial
import os
import numpy


ser = serial.Serial('/dev/cu.usbmodem21401', 115200)
cap = cv2.VideoCapture(0)

lastWord = ""
def found(name):
    name = name
    global lastWord
    if name != lastWord:
        print(name)
        ser.write(name.encode())
        lastWord = name
    time.sleep(1)

def start():
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
      # 0 is the ID of the default camera
    cv2.namedWindow("detection", cv2.WINDOW_AUTOSIZE)
    while True:  # Read a frame from the camera stream
        ret, frame = cap.read()
        if not ret:
            break
        labelDict = {}
        biggerConfidence = 0
          # Run YOLOv5 on the frame
        results = model(frame)
        detections = results.pandas().xyxy[0]  # Get a pandas DataFrame of the detected objects
        for _, row in detections.iterrows():
            label = row['name']
            labelDict[row['confidence']] = label
            if row['confidence'] > biggerConfidence:
                biggerConfidence = row['confidence']
            xmin, ymin, xmax, ymax = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            if xmin >= 0 and ymin >= 0 and xmax >= 0 and ymax >= 0:
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 255), 2)
                cv2.putText(frame, label, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 1)

                # Show the image with the labels
                cv2.imshow('detection', frame)
                key = cv2.waitKey(5)
                if key == ord('q'):
                    exit(1)
        if biggerConfidence in labelDict:
            found(labelDict[biggerConfidence])

if os.name == 'main':
    print("started")
    start()