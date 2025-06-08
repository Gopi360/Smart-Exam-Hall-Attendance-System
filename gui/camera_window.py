import tkinter as tk
from tkinter import messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
from facenet_pytorch import InceptionResnetV1, MTCNN
from scipy.spatial.distance import cosine
import torch
import time
from datetime import datetime
import os
from app.database import get_connection as get_db_connection
from app.utils import extract_roll_name
from gui.login_window import open_login_window

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(image_size=160, margin=0, device=device)
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

test_embeddings = np.load("test_embeddings.npy", allow_pickle=True).item()

already_marked = set()
last_print_time = {}

def recognize_face(face_embedding, threshold=0.5):
    best_match = None
    best_score = 1.0
    for filename, ref_embedding in test_embeddings.items():
        score = cosine(face_embedding, ref_embedding)
        if score < best_score and score < threshold:
            best_match = filename
            best_score = score
    return best_match, best_score

def mark_attendance(roll, name):
    db = get_db_connection()
    cursor = db.cursor()
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
        db.commit()
        print(f"🟢 Marked attendance: {roll} - {name} at {now}")
    else:
        print(f"ℹ Already marked today: {roll} - {name}")

    cursor.close()
    db.close()

def open_camera_window():
    cam_window = tk.Toplevel()
    cam_window.title("Face Recognition Attendance")
    cam_window.attributes("-fullscreen", True)

    label = tk.Label(cam_window)
    label.pack()

    btn_frame = tk.Frame(cam_window, bg='black')
    btn_frame.pack(side=tk.BOTTOM, fill=tk.X)

    def back_to_main():
        cap.release()
        cam_window.destroy()
        root = tk.Tk()
        root.withdraw()
        open_login_window(root)

    back_button = tk.Button(btn_frame, text="⬅ Back to Admin", command=back_to_main, font=("Arial", 12), bg="#E57373", fg="white")
    back_button.pack(pady=10)

    cap = cv2.VideoCapture(0)

    def update_frame():
        ret, frame = cap.read()
        if not ret:
            return

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
                        roll, name = extract_roll_name(match_name)
                        if roll and roll not in already_marked:
                            mark_attendance(roll, name)
                            already_marked.add(roll)

                        color = (0, 255, 0)
                        text = f"{name} ({score:.3f})"
                    else:
                        text = "Unknown"
                        color = (0, 0, 255)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        img_tk = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        label.imgtk = img_tk
        label.configure(image=img_tk)
        cam_window.after(10, update_frame)

    update_frame()
