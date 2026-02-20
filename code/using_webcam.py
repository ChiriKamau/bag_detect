import cv2
import time
import os
from datetime import datetime

save_directory = "/home/test/webcam_images"
delay_time = 30

if not os.path.exists(save_directory):
    os.makedirs(save_directory)

print(f"Waiting for {delay_time} seconds...")
time.sleep(delay_time)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not access webcam.")
    exit()

ret, frame = cap.read()

if ret:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{save_directory}/image_{timestamp}.jpg"
    cv2.imwrite(filename, frame)
    print(f"Image saved at: {filename}")
else:
    print("Failed to capture image.")

cap.release()