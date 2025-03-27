import cv2
import pickle
import os
import numpy as np

# Create a directory to store data if it doesn't exist
if not os.path.exists('data/'):
    os.mkdir('data/')

# Initialize webcam
video = cv2.VideoCapture(0)

# Load the Haar Cascade for face detection
facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# List to store face data
faces_data = []
i = 0

# Input Aadhaar number to associate with data
name = input("Enter your Aadhaar number: ")

# Parameters for capturing frames
framesTotal = 51
captureAfterFrame = 2

while True:
    # Capture frame from video
    ret, frame = video.read()

    if not ret:
        print("Failed to capture video. Exiting...")
        break

    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    # Draw rectangles around detected faces and collect face data
    for (x, y, w, h) in faces:
        cropped_face = frame[y:y + h, x:x + w]
        resized_img = cv2.resize(cropped_face, (50, 50))  # Resize to 50x50
        if len(faces_data) <= framesTotal and i % captureAfterFrame == 0:
            faces_data.append(resized_img)
        i += 1
        cv2.putText(frame, str(len(faces_data)), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 1)

    cv2.imshow("frame", frame)
    k = cv2.waitKey(1)
    if k == ord('q') or len(faces_data) >= framesTotal:
        break

video.release()
cv2.destroyAllWindows()

# Convert list to numpy array
faces_data = np.asarray(faces_data)
faces_data = faces_data.reshape((framesTotal, -1))
print(f"Total faces captured: {len(faces_data)}")

# ✅ Load existing names.pkl data safely
names_file = "data/names.pkl"
faces_file = "data/faces_data.pkl"

if os.path.exists(names_file):
    with open(names_file, "rb") as f:
        names = pickle.load(f)
    names.extend([name] * framesTotal)  # Append new data
else:
    names = [name] * framesTotal

# Save updated names
with open(names_file, "wb") as f:
    pickle.dump(names, f)

# ✅ Load existing faces_data.pkl safely
if os.path.exists(faces_file):
    with open(faces_file, "rb") as f:
        faces = pickle.load(f)
    faces = np.append(faces, faces_data, axis=0)  # Append new face data
else:
    faces = faces_data

# Save updated face data
with open(faces_file, "wb") as f:
    pickle.dump(faces, f)

print("✅ Face data and names successfully saved!")
