import cv2
import torch
import numpy as np
from PIL import Image
from datetime import datetime
import time
from facenet_pytorch import InceptionResnetV1, MTCNN
from scipy.spatial.distance import cosine
from app.database import get_connection

def recognize_and_mark_attendance(embedding_path="test_embeddings.npy"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mtcnn = MTCNN(image_size=160, margin=0, device=device)
    model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    test_embeddings = np.load(embedding_path, allow_pickle=True).item()

    conn = get_connection()
    cursor = conn.cursor()
    already_marked = set()
    last_print_time = {}

    cap = cv2.VideoCapture(0)
    print("📷 Camera started. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        face = mtcnn(img)

        if face is not None:
            face = face.unsqueeze(0).to(device)
            embedding = model(face).detach().cpu().numpy()[0]

            match_name, score = None, 1.0
            for filename, ref_embedding in test_embeddings.items():
                dist = cosine(embedding, ref_embedding)
                if dist < score and dist < 0.5:  # Threshold for matching
                    match_name = filename
                    score = dist

            boxes, _ = mtcnn.detect(img)
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = [int(b) for b in box]

                    if match_name:
                        roll, name = match_name.split("_", 1)[0], match_name.split("_", 1)[1].rsplit(".", 1)[0]
                        current_time = time.time()

                        if roll not in already_marked:
                            now = datetime.now()
                            cursor.execute(
                                "INSERT INTO attendance (roll, name, attendance_time) VALUES (%s, %s, %s)",
                                (roll, name, now)
                            )
                            conn.commit()
                            already_marked.add(roll)

                        text = f"{name} ({score:.3f})"
                        color = (0, 255, 0)
                    else:
                        text = "Unknown"
                        color = (0, 0, 255)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow("Face Recognition Attendance", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    cursor.close()
    conn.close()
