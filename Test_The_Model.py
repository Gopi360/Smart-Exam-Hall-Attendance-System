import cv2
import torch
import numpy as np
from PIL import Image
from datetime import datetime
import os
import time
from facenet_pytorch import InceptionResnetV1, MTCNN
from scipy.spatial.distance import cosine
import mysql.connector

# Device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(image_size=160, margin=0, device=device)
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Load known embeddings
test_embeddings = np.load("test_embeddings.npy", allow_pickle=True).item()

# MySQL connection setup
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="Gopi@007",
    database="Attendance_System"
)
cursor = conn.cursor()

def mark_attendance(roll, name):
    today = datetime.now().date()
    cursor.execute(
        "SELECT * FROM attendance WHERE roll = %s AND DATE(attendance_time) = %s",
        (roll, today)
    )
    result = cursor.fetchone()

    if not result:
        now = datetime.now()
        cursor.execute(
            "INSERT INTO attendance (roll, name, attendance_time) VALUES (%s, %s, %s)",
            (roll, name, now)
        )
        conn.commit()
        print(f"🟢 Marked attendance: {roll} - {name} at {now}")
    else:
        print(f"ℹ Already marked today: {roll} - {name}")


def recognize_face(face_embedding, threshold=0.5):
    best_match = None
    best_score = 1.0
    for filename, ref_embedding in test_embeddings.items():
        score = cosine(face_embedding, ref_embedding)
        if score < best_score and score < threshold:
            best_match = filename
            best_score = score
    return best_match, best_score

# Track who has already been marked
already_marked = set()
last_print_time = {}

# Start webcam
cap = cv2.VideoCapture(0)
print(" Starting camera. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    face = mtcnn(img)

    if face is not None:
        face = face.unsqueeze(0).to(device)
        embedding = model(face).detach().cpu().numpy()[0]

        match_name, score = recognize_face(embedding)

        boxes, _ = mtcnn.detect(img)
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = [int(b) for b in box]

                if match_name:
                    roll_name = match_name.split('.')[0]
                    if '_' in roll_name:
                        roll, name = roll_name.split('_', 1)
                    else:
                        roll, name = "Unknown", roll_name

                    current_time = time.time()

                    if roll not in already_marked:
                        mark_attendance(roll, name)
                        already_marked.add(roll)

                    if roll not in last_print_time or (current_time - last_print_time[roll]) > 10:
                        print(f"✅ Match: {roll} - {name} (Score: {score:.3f})")
                        last_print_time[roll] = current_time

                    color = (0, 255, 0)
                    text = f"{name} ({score:.3f})"
                else:
                    text = "Unknown"
                    color = (0, 0, 255)
                    print("❌ No match found.")


                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Face Recognition Attendance", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Close MySQL connection
cursor.close()
conn.close()
print(" Program finished. Camera closed.")