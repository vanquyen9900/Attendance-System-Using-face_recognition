import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import pandas as pd
from datetime import datetime
import main_code
import os
import csv


class AttendanceGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Điểm Danh Lớp Học")
        self.root.geometry("1000x600")
        self.root.configure(bg="white")
        self.last_message = ""

        # --- Time and Title ---
        self.time_label = tk.Label(root, text="", font=("Arial", 20, "bold"), fg="red", bg="white")
        self.time_label.pack(pady=10)
        self.update_time()

        frame_main = tk.Frame(root, bg="white")
        frame_main.pack(fill=tk.BOTH, expand=True)

        # --- Left frame (Notification + Camera + Exit) ---
        frame_left = tk.Frame(frame_main, bg="#87CEFA", bd=2, relief=tk.GROOVE)
        frame_left.place(x=10, y=10, width=500, height=500)

        tk.Label(frame_left, text="Điểm Danh Lớp Học", font=("Arial", 16, "bold"), fg="white", bg="blue").pack(fill=tk.X)

        tk.Label(frame_left, text="Thông báo", font=("Arial", 12, "bold"), bg="#87CEFA").pack(pady=(5, 0))
        self.lbl_notify = tk.Label(
            frame_left, text="", font=("Arial", 11), fg="black", bg="white",
            wraplength=470, justify="left", height=2, anchor="nw",
            relief="solid", padx=6, pady=4
        )
        self.lbl_notify.pack(pady=3)

        self.lbl_camera = tk.Label(frame_left, bg="white", width=400, height=300)
        self.lbl_camera.pack(pady=5)

        tk.Button(frame_left, text="Thoát", font=("Arial", 13), bg="orange", fg="black", command=self.quit_app)\
            .pack(pady=5)

        # --- Right frame (Add New Data + Attendance History) ---
        frame_right = tk.Frame(frame_main, bg="#87CEFA", bd=2, relief=tk.GROOVE)
        frame_right.place(x=510, y=10, width=480, height=540)

        tk.Label(frame_right, text="Thêm Dữ Liệu Mới", font=("Arial", 16, "bold"), fg="white", bg="blue").pack(fill=tk.X)

        tk.Label(frame_right, text="Nhập ID", font=("Arial", 12, "bold"), bg="#87CEFA").place(x=20, y=60)
        self.entry_id = tk.Entry(frame_right, font=("Arial", 12))
        self.entry_id.place(x=150, y=60, width=200)
        tk.Button(frame_right, text="Xoá", font=("Arial", 10), bg="orange", command=self.clear_id).place(x=370, y=58)

        tk.Label(frame_right, text="Nhập họ và tên", font=("Arial", 12, "bold"), bg="#87CEFA").place(x=20, y=120)
        self.entry_name = tk.Entry(frame_right, font=("Arial", 12))
        self.entry_name.place(x=150, y=120, width=200)
        tk.Button(frame_right, text="Xoá", font=("Arial", 10), bg="orange", command=self.clear_name).place(x=370, y=118)

        # Gộp lại thành 1 nút duy nhất
        tk.Button(frame_right, text="Lưu học viên mới", font=("Arial", 14, "bold"),
                  bg="lime", command=self.save_new_student).place(x=150, y=180, width=200, height=40)

        tk.Label(frame_right, text="Lịch sử điểm danh", font=("Arial", 14, "bold"), bg="#87CEFA").place(x=150, y=250)

        self.tree = ttk.Treeview(frame_right, columns=("Date", "ID", "Name", "Time"), show="headings", height=6)
        self.tree.place(x=20, y=280, width=440, height=200)
        for col in ("Date", "ID", "Name", "Time"):
            self.tree.heading(col, text=col)
            self.tree.column(col, anchor="center")

        self.load_attendance_history()

        # --- Start camera preview ---
        self.encodeList, self.classNames = main_code.get_encoded_data()
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        self.update_frame()

    def update_time(self):
        now = datetime.now()
        self.time_label.config(text=now.strftime("%d/%m/%Y   %H:%M:%S"))
        self.root.after(1000, self.update_time)

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame, message = main_code.process_frame(frame, self.encodeList, self.classNames)
            frame_resized = cv2.resize(frame, (400, 300))
            img = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            imgtk = ImageTk.PhotoImage(image=img)
            self.lbl_camera.imgtk = imgtk
            self.lbl_camera.configure(image=imgtk)

            if message and message.strip() and message != self.last_message:
                self.lbl_notify.config(text=message)
                self.last_message = message
                self.load_attendance_history()

        self.root.after(10, self.update_frame)

    def clear_id(self):
        self.entry_id.delete(0, tk.END)

    def clear_name(self):
        self.entry_name.delete(0, tk.END)
        
    def save_new_student(self):
        id_val = self.entry_id.get().strip()
        name_val = self.entry_name.get().strip()

        if not id_val or not name_val:
            messagebox.showwarning("Thiếu thông tin", "Vui lòng nhập đủ ID và Họ tên")
            return

        image_path = f"imagesAttendence/{id_val}.jpg"
        if os.path.exists(image_path):
            self.lbl_notify.config(text=f"Ảnh của ID {id_val} đã tồn tại.")
            return

        if os.path.exists("StudentsList.csv"):
            with open("StudentsList.csv", "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row["ID"] == id_val:
                        self.lbl_notify.config(text=f"ID {id_val} đã tồn tại trong danh sách.")
                        return

        ret, frame = self.cap.read()
        if ret:
            cv2.imwrite(image_path, frame)
        else:
            self.lbl_notify.config(text="Không thể chụp ảnh từ webcam.")
            return

        if not os.path.exists("StudentsList.csv"):
            with open("StudentsList.csv", "w", newline='', encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["ID", "Name"])

        with open("StudentsList.csv", "a", newline='', encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([id_val, name_val])

        self.lbl_notify.config(text=f"Đã lưu học viên mới: {id_val} - {name_val}")

        self.encodeList, self.classNames = main_code.get_encoded_data()


    def load_attendance_history(self):
        try:
            self.tree.delete(*self.tree.get_children())
            df = pd.read_csv("Attendance_Record.csv")
            for _, row in df.tail(6).iterrows():
                self.tree.insert("", "end", values=(row.Date, row.ID, row.Name, row.Time))
        except FileNotFoundError:
            self.lbl_notify.config(text="Không tìm thấy file Attendance_Record.csv")

    def quit_app(self):
        self.cap.release()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = AttendanceGUI(root)
    root.mainloop()
