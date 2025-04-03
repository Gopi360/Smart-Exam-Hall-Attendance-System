# import numpy as np
# import cv2
# import os
# import matplotlib.pyplot as plt
# from mtcnn import MTCNN
# from keras_facenet import FaceNet
# from scipy.spatial.distance import cosine
# # Load stored prototypes
# lfw_prototypes = np.load("E:\MCA_MAJOR_PROJECT\Smart-Exam-Hall-Attendance-System\lfw_prototypes.npy", allow_pickle=True).item()

# # Load test embeddings
# test_embeddings = np.load("E:\MCA_MAJOR_PROJECT\Smart-Exam-Hall-Attendance-System\lfw_embeddings.npy")
# test_names = np.load("E:\MCA_MAJOR_PROJECT\Smart-Exam-Hall-Attendance-System\lfw_names.npy")

# # Function to match a face with LFW dataset
# def match_face(embedding, prototypes, threshold=0.5):
#     best_match = None
#     best_score = 1.0  # Start with worst similarity

#     for name, prototype_embedding in prototypes.items():
#         score = cosine(embedding, prototype_embedding)
#         if score < best_score and score < threshold:
#             best_match = name
#             best_score = score

#     return best_match, best_score

# # Check each new test face
# for i, test_embedding in enumerate(test_embeddings):
#     name = test_names[i]
#     matched_name, match_score = match_face(test_embedding, lfw_prototypes)

#     if matched_name:
#         print(f"✅ {name} matched with {matched_name} (Score: {match_score:.2f})")
#     else:
#         print(f"❌ {name} was NOT found in LFW dataset")
