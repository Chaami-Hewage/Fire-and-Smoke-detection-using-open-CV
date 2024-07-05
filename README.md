# Fire-and-Smoke-detection-using-open-CV
Webcam Video Capture using OpenCV
This repository contains a simple script to capture video from a webcam and display it in a window using OpenCV.

Requirements
Python 3.x
OpenCV
You can install OpenCV using pip:


pip install opencv-python
Usage
Run the script to start capturing video from your default webcam and display it in a window. Press 'q' to quit the video stream.


import cv2

# Create a VideoCapture object to access the webcam
cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # Check if the frame was received successfully
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Display the captured frame in a window named 'webcam'
    cv2.imshow('webcam', frame)

    # Check if the 'q' key is pressed to quit the loop
    if cv2.waitKey(1) == ord('q'):
        break

# Release the VideoCapture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
Explanation
This code captures video from a webcam and displays it in real-time. Here’s a breakdown of what each part does:

Import the OpenCV library:

import cv2
Create a VideoCapture object to access the webcam:

cap = cv2.VideoCapture(0)
The 0 indicates the default webcam. If you have multiple cameras, you can change this index to access different ones.

Start an infinite loop to continuously capture frames:

while(True):
Capture a frame from the webcam:

ret, frame = cap.read()
The ret variable is a boolean indicating if the frame was successfully captured. The frame variable contains the captured image.

Check if the frame was received successfully:

if not ret:
    print("Can't receive frame (stream end?). Exiting ...")
    break
If ret is False, it means no frame was received (possibly because the stream has ended), so the loop is exited.

Display the captured frame in a window named 'webcam':

cv2.imshow('webcam', frame)
Check if the 'q' key is pressed to quit the loop:

if cv2.waitKey(1) == ord('q'):
    break
The cv2.waitKey(1) function waits for 1 millisecond for a key press. If the 'q' key is pressed, the loop breaks, stopping the video capture.

Release the VideoCapture object and close all OpenCV windows:

cap.release()
cv2.destroyAllWindows()
This ensures that the webcam is properly released and all windows are closed when the program ends.
