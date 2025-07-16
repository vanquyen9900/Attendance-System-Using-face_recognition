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


tracked_faces = []
lk_params = dict(winSize=(15, 15), maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

def process_frame(frame, encodeListKnown, classNames):
    global tracked_faces
    notify_message = ""

    flipped = cv2.flip(frame, 1)
    gray = cv2.cvtColor(flipped, cv2.COLOR_BGR2GRAY)
    rgb_small_frame = cv2.resize(flipped, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(rgb_small_frame, cv2.COLOR_BGR2RGB)

    now = datetime.now()
    new_tracked_faces = []

    if tracked_faces:
        prev_gray = process_frame.prev_gray if hasattr(process_frame, 'prev_gray') else gray
        for face in tracked_faces:
            new_points, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, face['points'], None, **lk_params)
            if new_points is not None and st.sum() > 3:
                for p in new_points:
                    x, y = p.ravel()
                    cv2.circle(flipped, (int(x), int(y)), 3, (0, 255, 0), -1)
                face['points'] = new_points
                face['last_seen'] = now
                new_tracked_faces.append(face)

                if new_points is not None and new_points.shape[0] > 0:
                    x, y = new_points[0].ravel()
                    x = int(x)
                    y = int(y)
                    cv2.putText(flipped, face['id'], (x-35, y - 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    process_frame.prev_gray = gray.copy()
    tracked_faces = new_tracked_faces

    if not tracked_faces:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        for face_encoding, face_location in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(encodeListKnown, face_encoding)
            face_distances = face_recognition.face_distance(encodeListKnown, face_encoding)
            match_index = np.argmin(face_distances)

            y1, x2, y2, x1 = [v * 4 for v in face_location]

            if face_distances[match_index] < 0.4 and matches[match_index]:
                id = classNames[match_index].upper()
                notify_message = f"Đã điểm danh: {id}"
                markAttendence(id)

                face_roi = gray[y1:y2, x1:x2]
                corners = cv2.goodFeaturesToTrack(face_roi, maxCorners=10, qualityLevel=0.3, minDistance=7)
                if corners is not None:
                    corners += np.array([[x1, y1]], dtype=np.float32)
                    tracked_faces.append({
                        'id': id,
                        'points': corners,
                        'last_seen': now
                    })

                cv2.rectangle(flipped, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(flipped, id, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            else:
                cv2.rectangle(flipped, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(flipped, "Unknown", (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                notify_message = "Không nhận diện được."

    if not tracked_faces:
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
