import time
import os
from picamera2 import Picamera2
from datetime import datetime

# ==============================
# USER CONFIGURATION
# ==============================

# Change this to your desired save directory
save_directory = "/home/test/test_bag" 

# Optional: custom filename (leave None for timestamped name)
custom_filename = None

# Delay time in seconds (2 minutes = 120 seconds)
delay_time = 120

# ==============================

# Ensure directory exists
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

print(f"Waiting for {delay_time} seconds before capturing image...")
time.sleep(delay_time)

# Initialize camera
picam2 = Picamera2()
picam2.configure(picam2.create_still_configuration())
picam2.start()

# Generate filename
if custom_filename:
    filename = custom_filename
else:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"image_{timestamp}.jpg"

file_path = os.path.join(save_directory, filename)

# Capture and save image
picam2.capture_file(file_path)

picam2.close()

print(f"Image saved successfully at: {file_path}")