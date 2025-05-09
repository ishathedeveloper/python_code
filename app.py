import os
import cv2
import joblib
import time
import numpy as np
import pandas as pd
import mysql.connector
from datetime import datetime, date
from flask import Flask, request, render_template, redirect, url_for
from sklearn.neighbors import KNeighborsClassifier

# FLASK APP
app = Flask(__name__)
MESSAGE = "WELCOME. Press 'a' to register your attendance"

# DATE FORMATTING
datetoday = date.today().strftime("%Y-%m-%d")
datetoday2 = date.today().strftime("%d-%B-%Y")

# VIDEO CAPTURE AND CASCADE
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

# MYSQL CONNECTION
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="Isha@2002",  # Your MySQL password
    database="sys"
)
cursor = db.cursor()

# DIRECTORY SETUP
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')


# UTILITIES
def totalreg():
    cursor.execute("SELECT COUNT(*) FROM users")
    return cursor.fetchone()[0]


def extract_faces(img):
    if img is not None:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return face_detector.detectMultiScale(gray, 1.3, 5)
    return []


def identify_face(facearray):
    model = joblib.load('static/face_recognition_model.pkl')
    return model.predict(facearray)


def train_model():
    faces, labels = [], []
    cursor.execute("SELECT face_folder FROM users")
    folders = cursor.fetchall()
    for folder in folders:
        user_folder = f'static/faces/{folder[0]}'
        for imgname in os.listdir(user_folder):
            img = cv2.imread(f'{user_folder}/{imgname}')
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(folder[0])
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(np.array(faces), labels)
    joblib.dump(knn, 'static/face_recognition_model.pkl')


def extract_attendance():
    cursor.execute("SELECT name, roll, time FROM attendance WHERE date = %s", (datetoday,))
    records = cursor.fetchall()
    names, rolls, times = zip(*records) if records else ([], [], [])
    return names, rolls, times, len(records)


def add_attendance(name):
    username, userid = name.split('_')
    current_time = datetime.now().strftime("%H:%M:%S")

    cursor.execute("SELECT * FROM attendance WHERE roll = %s AND date = %s", (userid, datetoday))
    if cursor.fetchone() is None:
        cursor.execute("INSERT INTO attendance (name, roll, time, date) VALUES (%s, %s, %s, %s)",
                       (username, userid, current_time, datetoday))
        db.commit()
    else:
        print("Already marked, but recording again")


# ROUTES
@app.route('/')
def home():
    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l,
                           totalreg=totalreg(), datetoday2=datetoday2, mess=MESSAGE)


@app.route('/start')
def start():
    if not os.path.isfile('static/face_recognition_model.pkl'):
        return render_template('home.html', mess='No trained model found. Please register first.',
                               totalreg=totalreg(), datetoday2=datetoday2)

    ATTENDENCE_MARKED = False
    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        for (x, y, w, h) in faces:
            face = cv2.resize(frame[y:y + h, x:x + w], (50, 50))
            identified_person = identify_face(face.reshape(1, -1))[0]
            cv2.putText(frame, f'{identified_person}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2)
            if cv2.waitKey(1) == ord('a'):
                add_attendance(identified_person)
                ATTENDENCE_MARKED = True
                break
        if ATTENDENCE_MARKED or cv2.waitKey(1) == ord('q'):
            break
        cv2.imshow('Press "a" to mark attendance, "q" to quit', frame)
    cap.release()
    cv2.destroyAllWindows()
    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l,
                           totalreg=totalreg(), datetoday2=datetoday2, mess='Attendance taken successfully')


@app.route('/add', methods=['POST'])
def add():
    newusername = request.form['newusername']
    newuserid = request.form['newuserid']
    face_folder = f"{newusername}_{newuserid}"
    userimagefolder = f'static/faces/{face_folder}'

    if not os.path.isdir(userimagefolder):
        os.makedirs(userimagefolder)

    cap = cv2.VideoCapture(0)
    i, j = 0, 0
    while True:
        _, frame = cap.read()
        faces = extract_faces(frame)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 20), 2)
            cv2.putText(frame, f'Images: {i}/50', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2)
            if j % 10 == 0:
                name = f'{newusername}_{i}.jpg'
                cv2.imwrite(f'{userimagefolder}/{name}', frame[y:y + h, x:x + w])
                i += 1
            j += 1
        if j == 500:
            break
        cv2.imshow('Adding new user', frame)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

    # Save to DB
    cursor.execute("INSERT INTO users (username, roll, face_folder) VALUES (%s, %s, %s)",
                   (newusername, newuserid, face_folder))
    db.commit()

    train_model()
    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l,
                           totalreg=totalreg(), datetoday2=datetoday2, mess='User added successfully')


if __name__ == '__main__':
    app.run(debug=True, port=1000)
