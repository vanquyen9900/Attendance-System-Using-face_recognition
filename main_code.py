import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import csv

path = "imagesAttendence"
images = []
classNames = []
myList = os.listdir(path)

for cls in myList:
    curImg = cv2.imread(f"{path}/{cls}")
    if curImg is not None:
        images.append(curImg)
        classNames.append(os.path.splitext(cls)[0])

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodes = face_recognition.face_encodings(img)
        if encodes:  
            encodeList.append(encodes[0])
    return encodeList

def markAttendence(id):
    id = str(id)
    info = []

    try:
        with open("StudentsList.csv", "r", encoding="utf-8") as file2:
            data = csv.DictReader(file2)
            for item in data:
                if item["ID"] == id:
                    info = [item["ID"], item["Name"]]
                    break
    except FileNotFoundError:
        return
    except KeyError:
        return

    if len(info) < 2:
        return

    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    timeStr = now.strftime("%H:%M:%S")
    cur = [date, id]

    try:
        with open("Attendance_Record.csv", "r+", encoding="utf-8") as file:
            lines = file.readlines()

            if not lines or not lines[0].startswith("Date,ID,Name,Time"):
                file.seek(0)
                file.write("Date,ID,Name,Time\n")
                lines = file.readlines()

            nameList = []
            for line in lines[1:]:
                entry = line.strip().split(",")
                if len(entry) >= 2:
                    nameList.append([entry[0], entry[1]])

            if cur not in nameList:
                file.write(f"{date},{id},{info[1]},{timeStr}\n")

    except FileNotFoundError:
        with open("Attendance_Record.csv", "w", encoding="utf-8") as file:
            file.write("Date,ID,Name,Time\n")
            file.write(f"{date},{id},{info[1]},{timeStr}\n")


def TakeImage(id, name):
    filename = f"{path}/{id}.jpg"
    if os.path.exists(filename):
        return False, f"Ảnh của ID {id} đã tồn tại."

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    for _ in range(5):  
        ret, frame = cap.read()

    if ret:
        cv2.imwrite(filename, frame)
        cap.release()
        return True, f" Đã lưu ảnh tại {filename}"
    else:
        cap.release()
        return False, "Không thể chụp ảnh từ webcam."


def SaveNewUserData(id, name):
    if not os.path.exists("StudentsList.csv"):
        with open("StudentsList.csv", "w", newline='', encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["ID", "Name"])

    with open("StudentsList.csv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["ID"] == id:
                return False, " ID đã tồn tại!"

    with open("StudentsList.csv", "a", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([id, name])
        return True, "Đã lưu thông tin học viên."

def process_frame(frame, encodeListKnown, classNames):
    notify_message = ""
    flipped = cv2.flip(frame, 1)
    small_frame = cv2.resize(flipped, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for face_encoding, face_location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(encodeListKnown, face_encoding)
        face_distances = face_recognition.face_distance(encodeListKnown, face_encoding)
        match_index = np.argmin(face_distances)

        y1, x2, y2, x1 = face_location
        y1 *= 4
        x2 *= 4
        y2 *= 4
        x1 *= 4

        if face_distances[match_index] < 0.45 and matches[match_index]:
            id = classNames[match_index].upper()
            cv2.rectangle(flipped, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(flipped, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(flipped, id, (x1 + 8, y2 - 8), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            markAttendence(id)
            notify_message = f"Đã điểm danh: {id}"
        else:
            cv2.rectangle(flipped, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.rectangle(flipped, (x1, y2 - 35), (x2, y2), (0, 0, 255), cv2.FILLED)
            cv2.putText(flipped, "Unknown", (x1 + 10, y2 - 10), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            notify_message = "Không nhận diện được."

    if not face_encodings:
            notify_message = "Không phát hiện khuôn mặt nào."

    return flipped, notify_message

def load_data():
    images = []
    classNames = []
    if not os.path.exists(path):
        os.makedirs(path)
    myList = os.listdir(path)
    for cls in myList:
        curImg = cv2.imread(f"{path}/{cls}")
        if curImg is not None:
            images.append(curImg)
            classNames.append(os.path.splitext(cls)[0])
    return images, classNames

def get_encoded_data():
    images, classNames = load_data()
    encodeList = findEncodings(images)
    return encodeList, classNames
