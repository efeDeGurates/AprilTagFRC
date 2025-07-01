import cv2
import numpy as np
from pyapriltags import Detector
import time
import serial
from networktables import NetworkTables
import socket

HOST = "192.168.x.x"
NetworkTables.initialize(server=HOST)
table = NetworkTables.getTable("apriltags")

#ser = serial.Serial('COMx', 115200, timeout=1)

time.sleep(2)

detector = Detector(families="tag36h11")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    tags = detector.detect(gray)

    for tag in tags:
        (top_left, top_right, bottom_right, bottom_left) = tag.corners

        center_x = int((top_left[0] + bottom_right[0]) / 2)
        center_y = int((top_left[1] + bottom_right[1]) / 2)

        print(f"Tag ID: {tag.tag_id} - X: {center_x}, Y: {center_y}")

        tag_table = table.getSubTable(f"tag_{tag.tag_id}")
        tag_table.putNumber("center_x", center_x)
        tag_table.putNumber("center_y", center_y)
        tag_table.putNumberArray("corners", np.array(tag.corners).flatten().tolist())
        tag_table.putNumber("last_update", time.time())

        #message = f"ID:{tag.tag_id}, X:{center_x}, Y:{center_y}\n"
        #ser.write(message.encode())

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
