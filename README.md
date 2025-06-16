# Smart Exam Hall Attendance System 🎓📷

A real-time face recognition-based attendance system designed to automate student verification in examination halls. This project reduces manual intervention, eliminates proxy attendance, and increases overall security — built using Python, Flask, OpenCV, FaceNet, and MTCNN.

---

## 🚀 Features

- ✅ **Web-based Registration Portal**  
  Students register with details and facial images via a Flask-based interface.

- 📸 **Real-time Face Detection & Recognition**  
  Uses MTCNN for face detection and FaceNet for generating embeddings.

- 🧠 **Automated Attendance Logging**  
  Matches registered students and stores timestamped attendance in a MySQL database.

- 🔒 **Admin Dashboard (GUI)**  
  Tkinter-based interface for:
  - Downloading students or attendance data (as Excel)
  - Deleting old records
  - Admin login authentication

- 🔁 **Embedding Generator Module**  
  Converts student images into FaceNet embeddings and stores them locally.

---

## 💻 Technologies Used

| Category              | Tech Stack                                   |
|-----------------------|----------------------------------------------|
| Programming Language  | Python 3.10                                  |
| Libraries             | OpenCV, facenet-pytorch, numpy, scipy, PIL   |
| Web Framework         | Flask (for registration)                     |
| GUI Toolkit           | Tkinter                                      |
| Face Recognition      | MTCNN (Detection), FaceNet (Recognition)     |
| Database              | MySQL (via mysql-connector-python)           |
| Development Tools     | VS Code, StarUML (for design), PyInstaller   |

<!-- ---

## 📂 Folder Structure Overview

Smart-Exam-Hall-Attendance-System/
├── app/ # Backend modules (embedding, recognition, DB)
├── gui/ # Admin GUI (Tkinter-based)
├── templates/ # HTML template for registration
├── static/ # Static assets like background images
├── Pictures/ # Student images (ROLL_NAME.jpg)
├── test_embeddings.npy
├── main.py # Launches GUI + recognition system
├── app.py # Flask app for registration
└── README.md -->


---

<!-- ## 📸 Screenshots

### 🔹 Registration Form
![Registration Form](screenshot/registration_form.png)

### 🔹 Real-time Attendance
![Face Recognition](screenshot/face_recognition1.png)
![Face Recognition](screenshot/face_recognition2.png)
![Face Recognition](screenshot/face_recognition3.png)
![Face Recognition](screenshot/face_recognition4.png)
![Face Recognition](screenshot/face_recognition5.png)
![Face Recognition](screenshot/face_recognition6.png)

### 🔹 Admin Panel
![Admin Panel](screenshot/admin_panel_1.png)
![Admin Panel](screenshot/admin_panel_2.png)

--- -->

## 📈 Future Scope

- 🔗 IoT Integration for smart alerts & access control  
- ☁️ Cloud support for centralized record access  
- 📱 Mobile app for admin monitoring  
- 🧑‍💼 Admin registration with role-based access


---

## 🙌 Acknowledgements

Special thanks to our guide Dr. Arijit Dey and teammates who supported us throughout this project.  
Built with passion and dedication as part of our MCA Major Project.

---

