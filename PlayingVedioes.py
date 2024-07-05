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
