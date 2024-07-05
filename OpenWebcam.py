import cv2

cap = cv2.VideoCapture(0)   
while(True):
    ret, frame = cap.read() # Capture frame-by-frame
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break     # Exit if no frame is received

    cv2.imshow('webcam', frame)
    if cv2.waitKey(1) == ord('q'):
        break      # Press 'q' to quit

cap.release() # Release the VideoCapture object
cv2.destroyAllWindows() # Close all windows
