import os
import numpy as np
from picamera2 import Picamera2
import tensorflow as tf
import cv2
from gpiozero import DigitalInputDevice
from time import sleep
from threading import Thread

import gpiod
import time
import atexit

relay1 = 14
relay2 = 15
chip = gpiod.Chip('gpiochip4')
chip2 = gpiod.Chip('gpiochip4')
led_line = chip.get_line(relay1)
led_line2 = chip2.get_line(relay2)
led_line.request(consumer="relay1", type=gpiod.LINE_REQ_DIR_OUT)
led_line2.request(consumer="relay2", type=gpiod.LINE_REQ_DIR_OUT)


# Initialize the Rain Sensor on Pin 17
rain_sensor = DigitalInputDevice(17)

picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": 'RGB888', "size": (640, 480)}))
picam2.start()

# Load the TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path="/home/rpi/Desktop/ENGR4950-Pedestrian-Warning/Old-TensorFlow-Project/model_unquant.tflite")
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']

# Load the list of labels
with open("/home/rpi/Desktop/ENGR4950-Pedestrian-Warning/Old-TensorFlow-Project/labels.txt", 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Start the camera
#cap = cv2.VideoCapture(0)
#im = picam2.capture_array()
#cap = cv2.imshow("Camera", im)
#im = picam2.capture_array()
#    cv2.imshow("Camera", im)
#    cv2.waitKey(1)
 
def exit_handler():
    led_line.set_value(0)
    led_line2.set_value(0)
    led_line.release()
    led_line2.release()

atexit.register(exit_handler)

def activateLightRelay():
    time_started = time.time()
    try:
        while not (time.time() > time_started + 5):
            led_line.set_value(1)
            time.sleep(0.5)
            led_line.set_value(0)
            led_line2.set_value(1)
            time.sleep(0.5)
            led_line2.set_value(0)
    finally:
        led_line.set_value(0)
        led_line2.set_value(0)



def readRainSensor():
    while True:
        if rain_sensor.is_active:
            print("no rain")
        else:
            print("rain")
        sleep(1)

def readCameraAndCompareModel():
    while True:
        # Capture frame-by-frame
        frame = picam2.capture_array()

        # Resize the frame to match the input tensor shape
        resized_frame = cv2.resize(frame, (input_shape[1], input_shape[2]))

        # Preprocess the image
        input_data = np.expand_dims(resized_frame, axis=0)
        input_data = (np.float32(input_data) - 127.5) / 127.5

        # Set the input tensor for the Interpreter
        interpreter.set_tensor(input_details[0]['index'], input_data)

        # Run interpreter
        interpreter.invoke()

        # Get the output tensor and label
        output_data = interpreter.get_tensor(output_details[0]['index'])
        label_index = np.argmax(output_data)

        # Display the label on the frame
        label = labels[label_index]
        cv2.putText(frame, label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


        # Display the resulting frame
        cv2.imshow('frame', frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        # Press 'p' to call Park()
        if cv2.waitKey(1) & 0xFF == ord('p'):
            break

        if label == "1 Cones":
            activateLightRelay()

# Initialize the Threads

rainThread = Thread(target = readRainSensor)
rainThread.setDaemon(True)

cameraThread = Thread(target = readCameraAndCompareModel)
cameraThread.setDaemon(True)

rainThread.start()
cameraThread.start()
while True:
    pass


# Release the camera and close all windows
#cap.release()
cv2.destroyAllWindows()
