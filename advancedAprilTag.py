import cv2
import numpy as np
from pyapriltags import Detector
import time
from networktables import NetworkTables

HOST = "192.168.x.x"
NetworkTables.initialize(server=HOST)

connected = False
def connection_listener(connected_, info):
    global connected
    connected = connected_
    print(f"NetworkTables connection status: {connected}")

NetworkTables.addConnectionListener(connection_listener, immediateNotify=True)

table = NetworkTables.getTable("apriltags")

detector = Detector(families="tag36h11")

cap = cv2.VideoCapture(0)

camera_matrix = np.array([[600, 0, 320],
                          [0, 600, 240],
                          [0, 0, 1]], dtype=np.float64)
dist_coeffs = np.zeros(5)

tag_size = 0.16

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    start_time = time.time()
    tags = detector.detect(gray)
    detect_time = time.time() - start_time

    for tag in tags:
        image_points = np.array([
            tag.corners[0],
            tag.corners[1],
            tag.corners[2],
            tag.corners[3]
        ], dtype=np.float64)

        half_size = tag_size / 2
        object_points = np.array([
            [-half_size,  half_size, 0],
            [ half_size,  half_size, 0],
            [ half_size, -half_size, 0],
            [-half_size, -half_size, 0]
        ], dtype=np.float64)

        success, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs)
        if not success:
            continue

        R, _ = cv2.Rodrigues(rvec)
        pose = np.eye(4)
        pose[:3, :3] = R
        pose[:3, 3] = tvec.flatten()

        trans = pose[:3, 3]
        dist = np.linalg.norm(trans)

        center_x = int(tag.center[0])
        center_y = int(tag.center[1])

        tag_table = table.getSubTable(f"tag_{tag.tag_id}")
        tag_table.putNumber("center_x", center_x)
        tag_table.putNumber("center_y", center_y)
        tag_table.putNumberArray("corners", np.array(tag.corners).flatten().tolist())
        tag_table.putNumber("distance", dist)
        tag_table.putNumberArray("pose", pose.flatten().tolist())
        tag_table.putNumber("last_update", time.time())
        print(f"Tag ID: {tag.tag_id} - Center: ({center_x}, {center_y}) - Distance: {dist:.3f} m")

    table.putNumber("detect_time", detect_time)
    table.putNumber("fps", 1.0 / detect_time if detect_time > 0 else 0)
    table.putBoolean("connected", connected)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
