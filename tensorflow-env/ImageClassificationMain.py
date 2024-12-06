# Toggle for some debug bits that may not end up in production
development = True

# cd Desktop/ENGR4950-Pedestrian-Warning/tensorflow-env/
# source bin/activate

# Picamera2 is how the raspberry pi interacts with the CSI (Camera Serial Interface) Camera
# Tensorflow is a machine learning model library that allows us to classify images
# cv2 allows us to handle the camera input from Picamera2
from picamera2 import Picamera2
import tensorflow as tf
import cv2
import numpy as np

# Gpiozero allows us to communicate through the gpi pins directly on the board, particularly for the rain sensor
from gpiozero import DigitalInputDevice
from time import sleep

# gpiod is another library that allows us to interact with the pins, particularly for the relay
import gpiod
import time
import atexit

# Threading allows us to run the camera and the rain sensor at the same time.
from threading import Thread

# Initializing the relay pins that will activate the lights.
# Pins 14 and 15
relay1 = 14
relay2 = 15

# Activate the pin lines, so we can directly access the pins
chip = gpiod.Chip('gpiochip4')
chip2 = gpiod.Chip('gpiochip4')
led_line = chip.get_line(relay1)
led_line2 = chip2.get_line(relay2)
led_line.request(consumer="relay1", type=gpiod.LINE_REQ_DIR_OUT)
led_line2.request(consumer="relay2", type=gpiod.LINE_REQ_DIR_OUT)

# Initialize the Rain Sensor on Pin 17
# This is all the setup it needs, thanks gpiozero!
rain_sensor = DigitalInputDevice(17)

# Initializing the camera, and creating a preview with a 640x480 size
# We only plan to do this in development, production will have no monitor and thus no need for a preview
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": 'RGB888', "size": (640, 480)}))
picam2.start()

# Load the TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path="/home/rpi/Desktop/ENGR4950-Pedestrian-Warning/tensorflow-env/model_unquant.tflite")
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']

# Load the list of labels
with open("/home/rpi/Desktop/ENGR4950-Pedestrian-Warning/tensorflow-env/labels.txt", 'r') as f:
    labels = [line.strip() for line in f.readlines()]
 
# Exit handler will release the relay before the program exits
def exit_handler():
    led_line.set_value(0)
    led_line2.set_value(0)
    led_line.release()
    led_line2.release()

# Register the handler so it actually does something
atexit.register(exit_handler)

# Light relay function, gets called by cameraThread (readCameraAndCompareModel)
# Checks time, then will alternate output to the relay for 30 seconds
# Then disables lights
def activateLightRelay():
    time_started = time.time()
    try:
        #while not (time.time() > time_started + 5):
        # First set of individual blinking
        led_line.set_value(1)
        time.sleep(0.5)
        led_line.set_value(0)
        time.sleep(0.5)
        led_line2.set_value(1)
        time.sleep(0.5)
        led_line2.set_value(0)
        time.sleep(0.5)
        # Second set of individual blinking
        led_line.set_value(1)
        time.sleep(0.5)
        led_line.set_value(0)
        time.sleep(0.5)
        led_line2.set_value(1)
        time.sleep(0.5)
        led_line2.set_value(0)
        time.sleep(0.5)
        # Both lights blinking set 1
        led_line.set_value(1)
        led_line2.set_value(1)
        time.sleep(0.5)
        led_line.set_value(0)
        led_line2.set_value(0)
        time.sleep(0.5)
        # Both lights blinking set 2
        led_line.set_value(1)
        led_line2.set_value(1)
        time.sleep(0.5)
        led_line.set_value(0)
        led_line2.set_value(0)
        time.sleep(2.5)
    finally:
        led_line.set_value(0)
        led_line2.set_value(0)

# Rain sensor function, gets spun into a thread
# checks if there is input on pin 17, if so
# activate lights (if its raining, drivers should be more careful)
def readRainSensor():
    while True:
        if rain_sensor.is_active:
            print("no rain")
        else:
            print("rain")
        sleep(15)

# Bulk of the operation
# captures a frame from picamera2, feeds it into the tensorflow to compare
# tensorflow spits out what it thinks is in the frame
# if statements at the end determine if lights should activate or not
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

        max_probability = int(np.max(output_data)*100)
        cv2.putText(frame, (str(max_probability)+"%"), (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


        # Display the resulting frame
        if development:
            cv2.imshow('frame', frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if (label == "1 Pedestrian" and max_probability>90):
            activateLightRelay()

        

# Initialize the Threads
#rainThread = Thread(target = readRainSensor, daemon = True)
cameraThread = Thread(target = readCameraAndCompareModel, daemon = True)


#rainThread.start()
cameraThread.start()
while True:
    pass
