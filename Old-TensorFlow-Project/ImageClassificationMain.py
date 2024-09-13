import os
import numpy as np
import tensorflow as tf
import cv2

# Load the TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path="/home/ethan/Desktop/CarProjectSubmission/TF/model_unquant.tflite")
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']

# Load the list of labels
with open("/home/ethan/Desktop/CarProjectSubmission/TF/labels.txt", 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Start the camera
cap = cv2.VideoCapture(0)
 
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

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

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
