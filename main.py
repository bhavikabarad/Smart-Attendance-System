import cv2
import numpy as np
import face_recognition
import mysql.connector
from datetime import datetime

# ------------------------
# Step 1: Connect to Database
# ------------------------
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="face_attendance"
)
cursor = conn.cursor(dictionary=True)
print("✅ Connected to MySQL Database!")

# ------------------------
# Step 2: Load Student Data
# ------------------------
cursor.execute("SELECT * FROM students")
students = cursor.fetchall()

images = []
classNames = []
studentIDs = []

for student in students:
    path = student["image_path"]
    img = cv2.imread(path)
    if img is not None:
        images.append(img)
        classNames.append(student["name"].upper())
        studentIDs.append(student["student_id"])
    else:
        print(f"⚠️ Could not read image for {student['name']} at {path}")

print("Face Recognition OK ✅")
print(f"Loaded {len(classNames)} students.")

# ------------------------
# Step 3: Encode Faces
# ------------------------
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodes = face_recognition.face_encodings(img)
        if encodes:
            encodeList.append(encodes[0])
    return encodeList

encodeListKnown = findEncodings(images)
print("Encoding Complete ✅")

# ------------------------
# Step 4: Mark Attendance & Log Entries
# ------------------------
def markAttendance(student_id, name):
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")

    # Step 4.1 — Check or insert daily attendance
    cursor.execute("SELECT * FROM attendance WHERE student_id = %s AND date = %s", (student_id, date_str))
    record = cursor.fetchone()

    if record:
        # If marked absent, update to present
        if record["status"] == "A":
            cursor.execute("""
                UPDATE attendance
                SET status = %s, in_time = %s
                WHERE student_id = %s AND date = %s
            """, ('P', time_str, student_id, date_str))
            conn.commit()
            print(f"✅ {name} marked Present at {time_str}")
        else:
            # Update only latest out_time (not every second)
            cursor.execute("""
                UPDATE attendance
                SET out_time = %s
                WHERE student_id = %s AND date = %s
            """, (time_str, student_id, date_str))
            conn.commit()
            print(f"🔁 {name} Out Time updated to {time_str}")
    else:
        # Create new record if none exists
        cursor.execute("""
            INSERT INTO attendance (student_id, date, status, in_time)
            VALUES (%s, %s, %s, %s)
        """, (student_id, date_str, 'P', time_str))
        conn.commit()
        print(f"🆕 Added {name} as Present at {time_str}")

    # Step 4.2 — Add tracking log (for all detections)
    cursor.execute("""
        INSERT INTO attendance_logs (student_id, date, time, action_type, recognized_by)
        VALUES (%s, %s, %s, %s, %s)
    """, (student_id, date_str, time_str, 'IN', 'Face'))
    conn.commit()

# ------------------------
# Step 5: Mark all students Absent by default (once per day)
# ------------------------
today = datetime.now().strftime("%Y-%m-%d")
cursor.execute("SELECT COUNT(*) AS count FROM attendance WHERE date = %s", (today,))
count = cursor.fetchone()["count"]

if count == 0:
    for sid in studentIDs:
        cursor.execute("""
            INSERT INTO attendance (student_id, date, status)
            VALUES (%s, %s, %s)
        """, (sid, today, 'A'))
    conn.commit()
    print("✅ All students marked Absent for today by default.")

# ------------------------
# Step 6: Start Webcam for Recognition
# ------------------------
cap = cv2.VideoCapture(0)
print("🎥 Press 'q' to quit camera window.")

while True:
    success, img = cap.read()
    if not success:
        print("❌ Camera not working properly.")
        break

    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex]
            student_id = studentIDs[matchIndex]

            y1, x2, y2, x1 = [v * 4 for v in faceLoc]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            markAttendance(student_id, name)

    cv2.imshow("Webcam", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
conn.close()
print("👋 Attendance System Closed.")
