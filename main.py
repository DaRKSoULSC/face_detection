import cv2

# Load the cascade(Pre-Trained Data)
face_cascade = cv2.CascadeClassifier('harcascade_frontalface_default.xml')

# To capture video from webcam.
webcam = cv2.VideoCapture(0)
# To use a video file as input
# webcam = cv2.VideoCapture('filename.mp4') for video file

while True:
    # Iterate over frame(Read frame)
    successful_frame_read, frame = webcam.read()
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect the faces
    face_coordinates = face_cascade.detectMultiScale(gray, 1.1, 4)
    # Draw rectangle around each face
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # Display
    cv2.imshow('Face Detector', frame)
    # Stop if escape key is pressed
    key = cv2.waitKey(30) & 0xff
    if key == 27:
        break
# Release the VideoCapture object
webcam.release()
