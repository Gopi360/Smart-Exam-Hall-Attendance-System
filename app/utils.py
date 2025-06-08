import os
from tkinter import messagebox

def ensure_dir_exists(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def confirm_action(prompt="Are you sure?"):
    """Ask user to confirm an action."""
    return messagebox.askyesno("Confirm", prompt)

def show_info(message):
    """Display info popup."""
    messagebox.showinfo("Info", message)

def show_error(message):
    """Display error popup."""
    messagebox.showerror("Error", message)

def extract_roll_name(filename):
    """Extract roll and name from filename format: Roll_Name.jpg"""
    if "_" in filename:
        roll = filename.split("_")[0]
        name = filename.split("_", 1)[1].rsplit(".", 1)[0]
        return roll, name
    return None, None