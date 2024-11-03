import cv2
import numpy as np

# Load the Haar Cascade face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Paths to your model files
model_path = 'mobilenet_iter_73000.caffemodel'  # Your model weights file
config_path = 'deploy.prototxt.txt'    # Your model config file

# Load the MobileNet SSD model
net = cv2.dnn.readNetFromCaffe(config_path, model_path)

# Define class labels for objects that MobileNet-SSD can detect
CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant",
    "sheep", "sofa", "train", "tvmonitor"
]

# Open the video file
video_path = 'video4.mp4'  # Replace with your actual video file path
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("End of video or cannot read frame.")
        break

    # Prepare the frame for object detection
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    # Process detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:  # Confidence threshold
            idx = int(detections[0, 0, i, 1])

            # If the detected object is a person
            if CLASSES[idx] == "person":
                # Get bounding box
                box = detections[0, 0, i, 3:7] * [frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]]
                (startX, startY, endX, endY) = box.astype("int")

                # Ensure bounding box is within frame dimensions
                startX = max(0, startX)
                startY = max(0, startY)
                endX = min(frame.shape[1], endX)
                endY = min(frame.shape[0], endY)

                # Check if the area is valid
                if endX > startX and endY > startY:
                    # Extract the person's area for face detection
                    person_area = frame[startY:endY, startX:endX]

                    # Convert to grayscale for face detection
                    gray_person_area = cv2.cvtColor(person_area, cv2.COLOR_BGR2GRAY)
                    # Detect faces in the person's area
                    faces = face_cascade.detectMultiScale(gray_person_area, scaleFactor=1.1, minNeighbors=5)

                    # Blur detected faces
                    for (fx, fy, fw, fh) in faces:
                        # Ensure the coordinates are within the bounds of the original frame
                        fx, fy, fw, fh = startX + fx, startY + fy, fw, fh
                        # Extract the face area
                        face_area = frame[fy:fy + fh, fx:fx + fw]
                        if face_area.size > 0:
                            blurred_face = cv2.GaussianBlur(face_area, (25, 25), 30)  # Adjust blur parameters as needed
                            frame[fy:fy + fh, fx:fx + fw] = blurred_face  # Replace the face area with the blurred version

                else:
                    print(f"Invalid bounding box coordinates: start=({startX}, {startY}), end=({endX}, {endY})")

    # Display the frame with the blurred face
    cv2.imshow("Blurred Face in Video", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
