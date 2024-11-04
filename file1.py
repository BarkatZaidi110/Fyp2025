import cv2
import numpy as np

# Load the Haar Cascades for frontal and profile face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')

# Open the video file
video_path = 'video.mp4'  # Replace with your actual video file path
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Initialize trackers list
trackers = []
tracker_type = 'KCF'  # KCF is faster but less precise with rotations

# Detection frequency
detection_interval = 15  # Detect every 15 frames
frame_count = 0

# Video properties for frame rate management
fps = cap.get(cv2.CAP_PROP_FPS)
frame_delay = int(1000 / fps) if fps > 0 else 1

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Reduce frame size for faster processing
    small_frame = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))
    gray_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)

    frame_count += 1

    # Run face detection every `detection_interval` frames
    if frame_count % detection_interval == 0:
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)
        profiles = profile_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

        detected_faces = list(faces) + list(profiles)

        # Clear previous trackers and add new ones for detected faces
        trackers = []
        for (x, y, w, h) in detected_faces:
            x, y, w, h = x * 2, y * 2, w * 2, h * 2  # Scale back to original frame size
            tracker = cv2.TrackerKCF_create()  # KCF tracker for faster performance
            tracker.init(frame, (x, y, w, h))
            trackers.append(tracker)

    # Update trackers and apply blur on tracked regions
    for tracker in trackers:
        success, bbox = tracker.update(frame)
        if success:
            x, y, w, h = [int(v) for v in bbox]
            face_area = frame[y:y + h, x:x + w]
            if face_area.size > 0:
                blurred_face = cv2.GaussianBlur(face_area, (21, 21), 15)
                frame[y:y + h, x:x + w] = blurred_face

    # Display the frame at the original frame rate
    cv2.imshow("Blurred Face in Video", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(frame_delay) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
