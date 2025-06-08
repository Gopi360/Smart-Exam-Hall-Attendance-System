import tkinter as tk
from tkinter import messagebox, simpledialog
import os
from app.database import export_students_to_excel, export_attendance_to_excel, delete_attendance_data, delete_students_data


def open_main_window():
    window = tk.Toplevel()
    window.title("Admin Dashboard")
    window.geometry("500x400")
    window.configure(bg="#f0f0f0")

    title = tk.Label(window, text="📋 Admin Panel - Attendance System", font=("Arial", 16, "bold"), bg="#f0f0f0")
    title.pack(pady=20)

    def handle_export_students():
        filename = simpledialog.askstring("File Name", "Enter filename to save student info (without extension):")
        if filename:
            filepath = os.path.join("exports", f"{filename}.xlsx")
            success = export_students_to_excel(filepath)
            if success:
                messagebox.showinfo("Success", f"Students info saved as {filepath}")
            else:
                messagebox.showerror("Error", "Failed to export student info.")

    def handle_export_attendance():
        filename = simpledialog.askstring("File Name", "Enter filename to save attendance info (without extension):")
        if filename:
            filepath = os.path.join("exports", f"{filename}.xlsx")
            success = export_attendance_to_excel(filepath)
            if success:
                messagebox.showinfo("Success", f"Attendance info saved as {filepath}")
            else:
                messagebox.showerror("Error", "Failed to export attendance info.")

    def handle_delete_attendance():
        confirm = simpledialog.askstring("Confirm Delete", "Type 'CONFIRM' to delete all attendance records:")
        if confirm == "CONFIRM":
            delete_attendance_data()
            messagebox.showinfo("Deleted", "All attendance data deleted.")

    def handle_delete_students():
        confirm = simpledialog.askstring("Confirm Delete", "Type 'CONFIRM' to delete all student records:")
        if confirm == "CONFIRM":
            delete_students_data()
            messagebox.showinfo("Deleted", "All student data deleted.")

    btn1 = tk.Button(window, text="📥 Download Students Info", font=("Arial", 12), width=30, command=handle_export_students)
    btn1.pack(pady=10)

    btn2 = tk.Button(window, text="📥 Download Attendance Info", font=("Arial", 12), width=30, command=handle_export_attendance)
    btn2.pack(pady=10)

    btn3 = tk.Button(window, text="🗑 Delete Attendance Data", font=("Arial", 12), width=30, fg="white", bg="red", command=handle_delete_attendance)
    btn3.pack(pady=10)

    btn4 = tk.Button(window, text="🗑 Delete Students Info", font=("Arial", 12), width=30, fg="white", bg="red", command=handle_delete_students)
    btn4.pack(pady=10)

    tk.Button(window, text="Logout", font=("Arial", 10), bg="gray", fg="white", command=window.destroy).pack(pady=20)


# if __name__ == '__main__':
#     root = tk.Tk()
#     root.withdraw()
#     open_main_window()
#     root.mainloop()
