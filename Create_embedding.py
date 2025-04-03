import cv2
import numpy as np
import os
from mtcnn import MTCNN
from keras_facenet import FaceNet

detector = MTCNN()
embedder = FaceNet()

test_dir = "E:\MCA_MAJOR_PROJECT\Smart-Exam-Hall-Attendance-System\Test"

test_embeddings = {}

for filename in os.listdir(test_dir):
    if filename.lower().endswith(('.png','.jpg', '.jpeg')):
        img_path = os.path.join(test_dir, filename)
        img = cv2.imread(img_path)
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        faces = detector.detect_faces(rgb_img)
        if faces:
            x, y, w, h = faces[0]['box']
            face = rgb_img[y:y+h, x:x+w]
            face = cv2.resize(face, (160, 160))
            face = np.expand_dims(face, axis=0)
            embedding = embedder.embeddings(face)[0]

            test_embeddings[filename] = embedding
            print(f"✅ Processed: {filename}")

np.save("test_embeddings.npy", test_embeddings)
print("✅ All embeddings saved!")
