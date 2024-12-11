import cv2
import face_recognition
import time

# Load the target image and encode the face
target_image = face_recognition.load_image_file("pic1.png")
target_encodings = face_recognition.face_encodings(target_image)

if len(target_encodings) == 0:
    print("Error: No face detected in the target image.")
    exit()

target_encoding = target_encodings[0]

# Load the video
video_path = "video2.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames.")

# Video writer to save the output
output_path = "optimized_output_video.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Processing optimizations
skip_frames = 2  # Skip every second frame to speed up
frame_count = 0
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Skip processing on certain frames for faster performance
    if frame_count % skip_frames != 0:
        out.write(frame)
        continue

    # Resize frame for faster face detection
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)  # 50% smaller
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Detect faces and encode
    face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces([target_encoding], face_encoding, tolerance=0.6)

        if matches[0]:
            # Adjust coordinates back to original frame size
            top *= 2
            right *= 2
            bottom *= 2
            left *= 2

            # Blur the face
            face_roi = frame[top:bottom, left:right]
            blurred_face = cv2.GaussianBlur(face_roi, (31, 31), 30)
            frame[top:bottom, left:right] = blurred_face

    # Write the processed frame to output
    out.write(frame)

    # Optional: Display video during processing
    cv2.imshow("Processing Video", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

end_time = time.time()
print(f"Processing complete in {end_time - start_time:.2f} seconds for {frame_count} frames.")
print("Output saved to:", output_path)

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
