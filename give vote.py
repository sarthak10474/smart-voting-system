from sklearn.neighbors import KNeighborsClassifier
import cv2
import pickle
import numpy as np
import os
import csv
import time
from datetime import datetime
import pyttsx3


def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()


video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


if not os.path.exists('data/'):
    os.mkdir('data/')


try:
    with open('data/names.pkl', 'rb') as f:
        LABELS = pickle.load(f)
    with open('data/faces_data.pkl', 'rb') as f:
        FACES = pickle.load(f)
    FACES = np.array(FACES).reshape(len(FACES), -1) 
except (EOFError, FileNotFoundError):
    print("Error: Face data or labels are missing!")
    exit()


knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS)


imgBackground = cv2.imread('background.png')
if imgBackground is None:
    print("Error: Background image not found!")
    exit()

bg_height, bg_width, _ = imgBackground.shape  
COL_NAMES = ['NAME', 'VOTE', 'DATE', 'TIME']


exists = os.path.isfile('votes.csv')


def check_if_exists(value):
    try:
        with open('votes.csv', 'r') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if row and row[0] == value:
                    return True
    except FileNotFoundError:
        return False
    return False

while True:
    ret, frame = video.read()
    if not ret:
        print("Error: Unable to capture video.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    output = None  

    for (x, y, w, h) in faces:
        crop_img = frame[y:y+h, x:x+w]
        resized_img = cv2.resize(crop_img, (50, 50)).reshape(1, -1) 

        output = knn.predict(resized_img)
        ts = time.time()
        date = datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
        timestamp = datetime.fromtimestamp(ts).strftime('%H:%M:%S')

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 1)
        cv2.putText(frame, str(output[0]), (x, y-15), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)

        attendance = [output[0], timestamp, date]

    
    frame_resized = cv2.resize(frame, (bg_width // 2, bg_height // 2))  
    resized_height, resized_width, _ = frame_resized.shape

    
    y_offset = bg_height - resized_height - 80  
    imgBackground[y_offset : y_offset + resized_height, 50 : 50 + resized_width] = frame_resized

    cv2.imshow('Face Voting System', imgBackground)

    k = cv2.waitKey(1)  

   
    if output is not None:
        if check_if_exists(output[0]):
            speak("You have already voted")
            break

        # ✅ Voting Options
        party = None
        if k == ord('1'):
            party = 'BJP'
        elif k == ord('2'):
            party = 'Congress'
        elif k == ord('3'):
            party = 'AAP'
        elif k == ord('4'):
            party = 'NOTA'

        # ✅ If a valid option is selected
        if party:
            speak(f"You have voted for {party}")
            time.sleep(3)

            # ✅ Save Vote to CSV
            with open('votes.csv', 'a' if exists else 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                if not exists:
                    writer.writerow(COL_NAMES)  # ✅ Write header only if file is new
                writer.writerow([output[0], party, date, timestamp])  # ✅ Save vote

            speak("Thank you for voting")
            break

video.release()
cv2.destroyAllWindows()
