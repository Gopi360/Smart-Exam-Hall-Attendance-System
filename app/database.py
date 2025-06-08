import mysql.connector
import pandas as pd

def get_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="Gopi@007",
        database="Attendance_System"
    )

def export_students_to_excel(filename):
    try:
        conn = get_connection()
        df = pd.read_sql("SELECT * FROM students_info", conn)
        df.to_excel(filename, index=False)
        conn.close()
        return True
    except Exception as e:
        print("❌ Error exporting students:", e)
        return False

def export_attendance_to_excel(filename):
    try:
        conn = get_connection()
        df = pd.read_sql("SELECT * FROM attendance", conn)
        df.to_excel(filename, index=False)
        conn.close()
        return True
    except Exception as e:
        print("❌ Error exporting attendance:", e)
        return False

def clear_students():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM students_info")
    conn.commit()
    cursor.close()
    conn.close()

def clear_attendance():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM attendance")
    conn.commit()
    cursor.close()
    conn.close()

def delete_attendance_data():
    clear_attendance()

def delete_students_data():
    clear_students()
