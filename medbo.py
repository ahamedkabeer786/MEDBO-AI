import tkinter as tk
import customtkinter as ctk
import random
import threading
import time
import pyttsx3
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation
from PIL import Image, ImageTk

# ----------------- Text-to-Speech ----------------- #
engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)  # Female voice
engine.setProperty('rate', 145)  # Slow and calm

def speak(text):
    engine.say(text)
    engine.runAndWait()

# ----------------- App Window ----------------- #
app = ctk.CTk()
app.title("MEDBO AI - Patient Monitoring System")
app.geometry("950x650")
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# ----------------- Variables ----------------- #
monitoring = False
heart_rate_data = []
spo2_data = []
bp_data = []

# ----------------- Top Frame with Logo ----------------- #
top_frame = ctk.CTkFrame(app)
top_frame.pack(pady=10)

# Load Medical Logo (Replace with your logo path)
logo_image = Image.open("medical_logo.png").resize((50, 50))
logo_photo = ImageTk.PhotoImage(logo_image)

logo_label = tk.Label(top_frame, image=logo_photo, bg="#1e1e1e")
logo_label.pack(side="left", padx=10)

title_label = ctk.CTkLabel(top_frame, text="🩺 MEDBO AI", font=("Arial", 28, "bold"), text_color="cyan")
title_label.pack(side="left", padx=10)

status_label = ctk.CTkLabel(app, text="Welcome to MEDBO AI", font=("Arial", 18))
status_label.pack(pady=5)

# Result Display
result_label = ctk.CTkLabel(app, text="", font=("Arial", 16))
result_label.pack(pady=10)

# ----------------- Graph Setup ----------------- #
fig, ax = plt.subplots(figsize=(6, 3))
ax.set_title("Live Patient Monitoring", color="white")
ax.set_facecolor("black")
fig.patch.set_facecolor("black")
ax.tick_params(colors="white")
ax.set_ylim(50, 150)  # Heart Rate range
line, = ax.plot([], [], color="lime", linewidth=2)
canvas = FigureCanvasTkAgg(fig, master=app)
canvas.get_tk_widget().pack(pady=20)

# ----------------- Animation Function ----------------- #
x_data = list(range(50))
y_data = [0] * 50

def animate(i):
    if monitoring:
        y_data.pop(0)
        heart_rate = random.randint(60, 120)  # HR
        y_data.append(heart_rate)
        line.set_data(range(len(y_data)), y_data)
        ax.set_xlim(0, len(y_data))
    return line,

ani = FuncAnimation(fig, animate, interval=500, blit=True)

# ----------------- Monitoring Logic ----------------- #
def start_monitoring():
    global monitoring
    monitoring = True
    status_label.configure(text="Monitoring Started...", text_color="green")
    threading.Thread(target=lambda: speak("Monitoring Started. Please stay calm."), daemon=True).start()
    threading.Thread(target=update_values, daemon=True).start()

def stop_monitoring():
    global monitoring
    monitoring = False
    status_label.configure(text="Monitoring Stopped", text_color="red")
    threading.Thread(target=lambda: speak("Monitoring Stopped."), daemon=True).start()

def update_values():
    while monitoring:
        heart_rate = random.randint(60, 120)
        spo2 = random.randint(94, 100)
        bp = f"{random.randint(100, 130)}/{random.randint(70, 90)}"

        condition = "Normal"
        if heart_rate < 60 or heart_rate > 100 or spo2 < 95:
            condition = "Alert ⚠️"
            threading.Thread(target=lambda: speak("Warning. Abnormal readings detected."), daemon=True).start()

        result_label.configure(
            text=f"Heart Rate: {heart_rate} bpm\nSpO₂: {spo2}%\nBP: {bp}\nStatus: {condition}",
            text_color="lime" if condition == "Normal" else "red"
        )

        time.sleep(2)

# ----------------- Buttons ----------------- #
button_frame = ctk.CTkFrame(app)
button_frame.pack(pady=10)

start_btn = ctk.CTkButton(button_frame, text="Start Monitoring", fg_color="green", hover_color="darkgreen", command=start_monitoring)
start_btn.grid(row=0, column=0, padx=20)

stop_btn = ctk.CTkButton(button_frame, text="Stop Monitoring", fg_color="red", hover_color="darkred", command=stop_monitoring)
stop_btn.grid(row=0, column=1, padx=20)

# ----------------- Delay Voice After UI Load ----------------- #
def welcome_message():
    time.sleep(1)  # Wait for UI to appear
    speak("Welcome to MEDBO AI, your advanced patient monitoring system.")

threading.Thread(target=welcome_message, daemon=True).start()

# ----------------- Run App ----------------- #
app.mainloop()
