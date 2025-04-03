import cv2
import numpy as np
from mtcnn import MTCNN
from keras_facenet import FaceNet
from scipy.spatial.distance import cosine

detector = MTCNN()
embedder = FaceNet()

test_embeddings = np.load("test_embeddings.npy", allow_pickle=True).item()

def recognize_face(face_embedding, threshold=0.5):
    best_match = None
    best_score = 1.0  

    for filename, ref_embedding in test_embeddings.items():
        score = cosine(face_embedding, ref_embedding)
        if score < best_score and score < threshold:
            best_match = filename
            best_score = score

    return best_match, best_score

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(rgb_frame)

    for face in faces:
        x, y, w, h = face['box']
        face_img = rgb_frame[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (160, 160))
        face_img = np.expand_dims(face_img, axis=0)
        face_embedding = embedder.embeddings(face_img)[0]

        matched_filename, match_score = recognize_face(face_embedding)

        if matched_filename:
            text = f"{matched_filename} ({match_score:.2f})"
            color = (0, 255, 0)
        else:
            text = "Unknown"
            color = (0, 0, 255)
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Real-Time Face Recognition", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
