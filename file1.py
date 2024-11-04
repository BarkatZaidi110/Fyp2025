import cv2
import numpy as np

# Load Haar Cascades for frontal and profile face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')

# Open the video file
video_path = 'video.mp4'  # Replace with your actual video file path
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# List to store individual trackers for each detected face
trackers = []
tracker_type = 'KCF'  # KCF is faster but may be less precise with rotations

# Detection frequency
detection_interval = 30  # Detect every 30 frames
frame_count = 0

# Video properties for frame rate management
fps = cap.get(cv2.CAP_PROP_FPS)
frame_delay = int(1000 / fps) if fps > 0 else 1

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    faces_detected = 0  # Count of faces detected in each frame
    blurred_faces = 0  # Count of faces successfully blurred in each frame

    # Only run face detection every `detection_interval` frames
    if frame_count % detection_interval == 0:
        # Resize frame for faster processing based on resolution
        scale_percent = 50  # Adjust based on video size
        width = int(frame.shape[1] * scale_percent / 100)
        height = int(frame.shape[0] * scale_percent / 100)
        small_frame = cv2.resize(frame, (width, height))
        gray_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)

        # Detect faces and profiles
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)
        profiles = profile_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

        detected_faces = list(faces) + list(profiles)
        faces_detected = len(detected_faces)

        # Clear previous trackers and add new ones for detected faces
        trackers = []
        for (x, y, w, h) in detected_faces:
            x, y, w, h = int(x * 2), int(y * 2), int(w * 2), int(h * 2)  # Scale to original frame size
            tracker = cv2.TrackerKCF_create()  # Create a KCF tracker
            tracker.init(frame, (x, y, w, h))
            trackers.append(tracker)

    # Update trackers and apply blur on tracked regions
    for idx, tracker in enumerate(trackers):
        success, bbox = tracker.update(frame)
        if success:
            x, y, w, h = [int(v) for v in bbox]
            face_area = frame[y:y + h, x:x + w]
            if face_area.size > 0:
                blurred_face = cv2.GaussianBlur(face_area, (21, 21), 15)
                frame[y:y + h, x:x + w] = blurred_face
                blurred_faces += 1
                print(f"Frame {frame_count}: Tracker {idx + 1} successfully blurred at position {x}, {y}, {w}, {h}.")
        else:
            print(f"Frame {frame_count}: Tracker {idx + 1} lost.")

    # Print a summary for the current frame
    print(f"Frame {frame_count}: {faces_detected} face(s) detected, {blurred_faces} face(s) blurred.")

    # Display the frame at the original frame rate
    cv2.imshow("Blurred Face in Video", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(frame_delay) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
