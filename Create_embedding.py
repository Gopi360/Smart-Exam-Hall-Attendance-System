import os
import torch
import numpy as np
from PIL import Image
from facenet_pytorch import InceptionResnetV1, MTCNN

# Initialize models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(image_size=160, margin=0, device=device)
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Directory for test images
test_dir = "E:/MCA_MAJOR_PROJECT/Smart-Exam-Hall-Attendance-System/Pictures"
test_embeddings = {}

for filename in os.listdir(test_dir):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        img_path = os.path.join(test_dir, filename)
        img = Image.open(img_path).convert('RGB')
        
        face = mtcnn(img)
        if face is not None:
            face = face.unsqueeze(0).to(device)
            embedding = model(face).detach().cpu().numpy()[0]
            test_embeddings[filename] = embedding
            print(f"✅ Processed: {filename}")
        else:
            print(f"⚠ Face not detected in: {filename}")

np.save("test_embeddings.npy", test_embeddings)
print("✅ All embeddings saved!")