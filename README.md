
# 📹 Webcam and Video Playback using OpenCV

This repository contains scripts to capture video from a webcam and play video files using OpenCV.

## 📋 Requirements

- Python 3.x
- OpenCV

You can install OpenCV using pip:
```bash
pip install opencv-python
```

## 🚀 Usage

### Webcam Video Capture

Run the script to start capturing video from your default webcam and display it in a window. Press 'q' to quit the video stream.

```python
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
```

### Video Playback

Run the script to play a video file. Press 'q' to quit the video playback.

```python
import cv2

def play_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return  # Print an error if video is not opened
    
    while True:
        ret, frame = cap.read()  # Capture frame-by-frame
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break   # Exit if no frame is received

        cv2.imshow('video', frame)  # Display the resulting frame
        if cv2.waitKey(25) == ord('q'):
            break   # Press 'q' to quit

    cap.release()  # Release the VideoCapture object
    cv2.destroyAllWindows()  # Close all windows

video_path = 'C:/Users/Asus/Desktop/Fire-and-Smoke-detection-using-open-CV/vedioes/fire1.mp4'
play_video(video_path)
```

## 📝 Explanation

### Webcam Video Capture

This code captures video from a webcam and displays it in real-time. Here’s a breakdown of what each part does:

1. **Import the OpenCV library:**
   ```python
   import cv2
   ```

2. **Create a VideoCapture object to access the webcam:**
   ```python
   cap = cv2.VideoCapture(0)
   ```
   The `0` indicates the default webcam. If you have multiple cameras, you can change this index to access different ones.

3. **Start an infinite loop to continuously capture frames:**
   ```python
   while(True):
   ```

4. **Capture a frame from the webcam:**
   ```python
   ret, frame = cap.read()
   ```
   The `ret` variable is a boolean indicating if the frame was successfully captured. The `frame` variable contains the captured image.

5. **Check if the frame was received successfully:**
   ```python
   if not ret:
       print("Can't receive frame (stream end?). Exiting ...")
       break
   ```
   If `ret` is `False`, it means no frame was received (possibly because the stream has ended), so the loop is exited.

6. **Display the captured frame in a window named 'webcam':**
   ```python
   cv2.imshow('webcam', frame)
   ```

7. **Check if the 'q' key is pressed to quit the loop:**
   ```python
   if cv2.waitKey(1) == ord('q'):
       break
   ```
   The `cv2.waitKey(1)` function waits for 1 millisecond for a key press. If the 'q' key is pressed, the loop breaks, stopping the video capture.

8. **Release the VideoCapture object and close all OpenCV windows:**
   ```python
   cap.release()
   cv2.destroyAllWindows()
   ```
   This ensures that the webcam is properly released and all windows are closed when the program ends.

### Video Playback

This code plays a video file and displays it in a window. Here’s a breakdown of what each part does:

1. **Import the OpenCV library:**
   ```python
   import cv2
   ```

2. **Define the `play_video` function to play a video file:**
   ```python
   def play_video(video_path):
   ```

3. **Create a VideoCapture object to access the video file:**
   ```python
   cap = cv2.VideoCapture(video_path)
   if not cap.isOpened():
       print("Error: Could not open video.")
       return  # Print an error if video is not opened
   ```

4. **Start an infinite loop to continuously capture frames:**
   ```python
   while True:
   ```

5. **Capture a frame from the video file:**
   ```python
   ret, frame = cap.read()  # Capture frame-by-frame
   if not ret:
       print("Can't receive frame (stream end?). Exiting ...")
       break   # Exit if no frame is received
   ```

6. **Display the captured frame in a window named 'video':**
   ```python
   cv2.imshow('video', frame)  # Display the resulting frame
   ```

7. **Check if the 'q' key is pressed to quit the loop:**
   ```python
   if cv2.waitKey(25) == ord('q'):
       break   # Press 'q' to quit
   ```

8. **Release the VideoCapture object and close all OpenCV windows:**
   ```python
   cap.release()  # Release the VideoCapture object
   cv2.destroyAllWindows()  # Close all windows
   ```

