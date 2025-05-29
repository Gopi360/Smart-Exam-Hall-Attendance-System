import os
from flask import Flask, render_template, request, redirect, url_for, flash
import mysql.connector
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = "E:/MCA_MAJOR_PROJECT/Smart-Exam-Hall-Attendance-System/Pictures"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="Gopi@007",
    database="Attendance_System"
)
cursor = db.cursor()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def register():
    message = None

    if request.method == 'POST':
        roll = request.form['roll']
        name = request.form['name']
        stream = request.form['stream']
        file = request.files['photo']

        # Check if student is already registered
        cursor.execute("SELECT * FROM students_info WHERE roll = %s", (roll,))
        existing_student = cursor.fetchone()

        if existing_student:
            message = "⚠️ Student is already registered."
        elif file and allowed_file(file.filename):
            filename = f"{roll}_{name.replace(' ', '_').upper()}.jpg"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            try:
                sql = "INSERT INTO students_info (roll, name, stream, photo_filename) VALUES (%s, %s, %s, %s)"
                val = (roll, name, stream, filename)
                cursor.execute(sql, val)
                db.commit()
                return redirect(url_for('register', success=1))
            except mysql.connector.Error as err:
                message = f"❌ Error saving data: {err}"
        else:
            message = "❌ Invalid file type!"

    if request.args.get('success'):
        message = "✅ Registered successfully!"

    return render_template("register.html", message=message)

if __name__ == '__main__':
    app.run(debug=True)