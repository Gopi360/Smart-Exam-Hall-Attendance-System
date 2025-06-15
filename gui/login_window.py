import tkinter as tk
from tkinter import messagebox
from gui.main_window import open_main_window

def open_login_window(root):
    login_window = tk.Toplevel(root)
    login_window.title("Admin Login")
    login_window.geometry("300x200")
    login_window.resizable(False, False)

    tk.Label(login_window, text="Username:").pack(pady=5)
    username_entry = tk.Entry(login_window)
    username_entry.pack(pady=5)

    tk.Label(login_window, text="Password:").pack(pady=5)
    password_entry = tk.Entry(login_window, show="*")
    password_entry.pack(pady=5)

    def validate_login():
        if username_entry.get() == "admin" and password_entry.get() == "admin":
            messagebox.showinfo("Login Success", "Welcome, Admin!")
            login_window.destroy()
            open_main_window()
        else:
            messagebox.showerror("Login Failed", "Invalid credentials.")

    tk.Button(login_window, text="Login", command=validate_login).pack(pady=10)
