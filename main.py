# import tkinter as tk
# from app.embedder import generate_embeddings
# from gui.main_window import open_main_window

# def main():
#     print("🔁 Generating face embeddings...")
#     generate_embeddings("Pictures/")
#     print("✅ Embeddings ready!")

#     root = tk.Tk()
#     root.withdraw()  # Hide the default root window
#     open_main_window()
#     root.mainloop()

# if __name__ == "__main__":
#     main()

import tkinter as tk
from app.embedder import generate_embeddings
from gui.camera_window import open_camera_window  # ✅ use camera instead of admin panel

def main():
    # print("🔁 Generating face embeddings...")
    # generate_embeddings("Pictures/")  # make sure this folder exists and has images
    # print("✅ Embeddings ready!")

    root = tk.Tk()
    root.withdraw()  # Hide the root window
    open_camera_window()  # ✅ open camera window instead of admin panel
    root.mainloop()

if __name__ == "__main__":
    main()
