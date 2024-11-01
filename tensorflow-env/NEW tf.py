import tensorflow as tf
import cv2
import numpy as np
from picamera2 import Picamera2

# Load the model
model = tf.lite.Interpreter(model_path="/home/rpi/Desktop/ENGR4950-Pedestrian-Warning/tensorflow-env/model_unquant.tflite")

# Load an image
picam2 = Picamera2()
image =  picam2.capture_array()
image_np = np.array(image)

# Run inference
input_tensor = tf.convert_to_tensor(image_np)
input_tensor = input_tensor[tf.newaxis, ...]
detections = model(input_tensor)

# Extract bounding box coordinates and draw them
for detection in detections['detection_boxes'][0]:
   ymin, xmin, ymax, xmax = detection.numpy()
   ymin, xmin, ymax, xmax = int(ymin * image.shape[0]), int(xmin * image.shape[1]), int(ymax * image.shape[0]), int(xmax * image.shape[1])
   cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

# Display the image
cv2.imshow('Image', image)
cv2.waitKey(0)

