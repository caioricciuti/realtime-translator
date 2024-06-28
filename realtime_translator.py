import sounddevice as sd
import numpy as np
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import whisper
import queue
import threading
import customtkinter as ctk
import torch
import datetime
import time
import os
from scipy import signal
from sklearn.cluster import KMeans
import psutil
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter.messagebox as tkmb

# Set appearance mode and color theme
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# Load models
whisper_model = whisper.load_model("base")
m2m_model = M2M100ForConditionalGeneration.from_pretrained(
    "facebook/m2m100_418M")
m2m_tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")

# Audio settings
RATE = 16000
CHUNK = int(RATE * 2)  # 2 seconds of audio
audio_queue = queue.Queue()

# Supported languages
LANGUAGES = {
    "English": "en", "German": "de", "French": "fr", "Spanish": "es",
    "Italian": "it", "Japanese": "ja", "Chinese": "zh", "Russian": "ru",
    "Portuguese": "pt", "Arabic": "ar", "Korean": "ko", "Dutch": "nl"
}


class TranslatorApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Real-Time Translator")
        self.geometry("900x700")

        # Initialize default values
        self.cpu_limit = 50  # Default CPU usage limit (%)
        self.memory_limit = 75  # Default memory usage limit (%)

        # Create tab view
        self.tab_view = ctk.CTkTabview(self)
        self.tab_view.pack(fill="both", expand=True, padx=20, pady=20)

        # Create tabs
        self.translation_tab = self.tab_view.add("Translation")
        self.logs_tab = self.tab_view.add("Logs")
        self.config_tab = self.tab_view.add("Config")
        self.resources_tab = self.tab_view.add("Resources")

        self.setup_translation_tab()
        self.setup_logs_tab()
        self.setup_config_tab()
        self.setup_resources_tab()

        self.is_translating = False
        self.stream = None
        self.processing_thread = None
        self.start_time = None
        self.speaker_kmeans = KMeans(n_clusters=3)  # Assuming max 3 speakers
        self.speaker_features = []

    def setup_translation_tab(self):
        frame = ctk.CTkFrame(self.translation_tab)
        frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Language selection
        self.source_lang_var = ctk.StringVar(value="German")
        self.target_lang_var = ctk.StringVar(value="English")

        source_lang_label = ctk.CTkLabel(frame, text="Source Language:")
        source_lang_label.pack(pady=(0, 5))
        self.source_lang_menu = ctk.CTkOptionMenu(frame, values=list(
            LANGUAGES.keys()), variable=self.source_lang_var)
        self.source_lang_menu.pack(pady=(0, 10))

        target_lang_label = ctk.CTkLabel(frame, text="Target Language:")
        target_lang_label.pack(pady=(0, 5))
        self.target_lang_menu = ctk.CTkOptionMenu(frame, values=list(
            LANGUAGES.keys()), variable=self.target_lang_var)
        self.target_lang_menu.pack(pady=(0, 10))

        # Transcription and translation display
        self.text_area = ctk.CTkTextbox(frame, height=300, width=700)
        self.text_area.pack(pady=10)

        # Start/Stop button
        self.toggle_button = ctk.CTkButton(
            frame, text="Start Translation", command=self.toggle_translation)
        self.toggle_button.pack(pady=10)

        # Elapsed time label
        self.elapsed_time_label = ctk.CTkLabel(
            frame, text="Elapsed Time: 00:00:00")
        self.elapsed_time_label.pack(pady=5)

    def setup_logs_tab(self):
        frame = ctk.CTkFrame(self.logs_tab)
        frame.pack(fill="both", expand=True, padx=10, pady=10)

        self.log_area = ctk.CTkTextbox(frame, height=500, width=700)
        self.log_area.pack(pady=10)

    def setup_config_tab(self):
        frame = ctk.CTkFrame(self.config_tab)
        frame.pack(fill="both", expand=True, padx=10, pady=10)

        cpu_label = ctk.CTkLabel(frame, text="CPU Usage Limit (%)")
        cpu_label.pack(pady=(0, 5))
        self.cpu_slider = ctk.CTkSlider(
            frame, from_=10, to=100, number_of_steps=90)
        self.cpu_slider.set(self.cpu_limit)
        self.cpu_slider.pack(pady=(0, 10))

        memory_label = ctk.CTkLabel(frame, text="Memory Usage Limit (%)")
        memory_label.pack(pady=(0, 5))
        self.memory_slider = ctk.CTkSlider(
            frame, from_=10, to=100, number_of_steps=90)
        self.memory_slider.set(self.memory_limit)
        self.memory_slider.pack(pady=(0, 10))

        # Additional resource info labels
        self.total_cpu_cores_label = ctk.CTkLabel(
            frame, text=f"Total CPU Cores: {psutil.cpu_count()}")
        self.total_cpu_cores_label.pack(pady=5)
        self.total_memory_label = ctk.CTkLabel(frame, text=f"Total Memory: {
                                               psutil.virtual_memory().total // (1024 ** 2)} MB")
        self.total_memory_label.pack(pady=5)
        self.available_memory_label = ctk.CTkLabel(frame, text=f"Available Memory: {
                                                   psutil.virtual_memory().available // (1024 ** 2)} MB")
        self.available_memory_label.pack(pady=5)
        self.disk_usage_label = ctk.CTkLabel(frame, text=f"Disk Usage: {
                                             psutil.disk_usage('/').percent}%")
        self.disk_usage_label.pack(pady=5)
        self.network_io_label = ctk.CTkLabel(frame, text=f"Network IO: Sent {psutil.net_io_counters(
        ).bytes_sent // (1024 ** 2)} MB, Received {psutil.net_io_counters().bytes_recv // (1024 ** 2)} MB")
        self.network_io_label.pack(pady=5)

        save_button = ctk.CTkButton(
            frame, text="Save Configuration", command=self.save_config)
        save_button.pack(pady=10)

    def setup_resources_tab(self):
        frame = ctk.CTkFrame(self.resources_tab)
        frame.pack(fill="both", expand=True, padx=10, pady=10)

        self.cpu_label = ctk.CTkLabel(frame, text="CPU Usage: 0%")
        self.cpu_label.pack(pady=5)
        self.memory_label = ctk.CTkLabel(frame, text="Memory Usage: 0%")
        self.memory_label.pack(pady=5)

        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(8, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=frame)
        self.canvas.get_tk_widget().pack(pady=10)

        self.cpu_data = [0] * 60
        self.memory_data = [0] * 60

    def save_config(self):
        self.cpu_limit = int(self.cpu_slider.get())
        self.memory_limit = int(self.memory_slider.get())
        self.log_message(f"Configuration saved: CPU limit {
                         self.cpu_limit}%, Memory limit {self.memory_limit}%")
        self.show_toast("Configuration saved successfully!")

    def toggle_translation(self):
        if not self.is_translating:
            self.start_translation()
        else:
            self.stop_translation()

    def start_translation(self):
        self.is_translating = True
        self.toggle_button.configure(text="Stop Translation")
        self.stream = sd.InputStream(
            callback=self.audio_callback, channels=1, samplerate=RATE, blocksize=CHUNK)
        self.stream.start()
        self.processing_thread = threading.Thread(target=self.process_audio)
        self.processing_thread.start()
        self.start_time = time.time()
        self.update_elapsed_time()
        self.update_resource_usage()

    def stop_translation(self):
        self.is_translating = False
        self.toggle_button.configure(text="Start Translation")
        if self.stream:
            self.stream.stop()
            self.stream.close()
        if self.processing_thread and self.processing_thread.is_alive():
            # Use timeout to avoid blocking
            self.processing_thread.join(timeout=1)
        self.ask_save_translation()

    def ask_save_translation(self):
        save_window = ctk.CTkToplevel(self)
        save_window.title("Save Translation")
        save_window.geometry("300x100")

        label = ctk.CTkLabel(
            save_window, text="Do you want to save the translation?")
        label.pack(pady=10)

        def save_translation():
            filename = f"translation_{
                datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            with open(filename, "w", encoding="utf-8") as f:
                f.write(self.text_area.get("1.0", "end"))
            self.log_message(f"Translation saved to {filename}")
            save_window.destroy()
            self.show_toast(f"Translation saved to {filename}")

        def close_window():
            save_window.destroy()

        save_button = ctk.CTkButton(
            save_window, text="Save", command=save_translation)
        save_button.pack(side="left", padx=10)
        cancel_button = ctk.CTkButton(
            save_window, text="Cancel", command=close_window)
        cancel_button.pack(side="right", padx=10)

    def audio_callback(self, indata, frames, time, status):
        if status:
            print(status)
        audio_queue.put(indata.copy())

    def process_audio(self):
        while self.is_translating:
            try:
                audio_data = audio_queue.get(timeout=1)
                result = whisper_model.transcribe(audio_data.flatten())
                transcribed_text = result["text"].strip()

                if transcribed_text:
                    source_lang = LANGUAGES[self.source_lang_var.get()]
                    target_lang = LANGUAGES[self.target_lang_var.get()]

                    # Identify speaker
                    speaker = self.identify_speaker(audio_data.flatten())

                    # Tokenize and translate
                    inputs = m2m_tokenizer(
                        transcribed_text, return_tensors="pt", padding=True)
                    translated = m2m_model.generate(
                        **inputs, forced_bos_token_id=m2m_tokenizer.get_lang_id(target_lang))
                    translated_text = m2m_tokenizer.decode(
                        translated[0], skip_special_tokens=True)

                    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    self.update_display(f"[{timestamp}] Speaker {speaker}:\n")
                    self.update_display(f"Source ({source_lang}): {
                                        transcribed_text}\n")
                    self.update_display(f"Translation ({target_lang}): {
                                        translated_text}\n\n")

            except queue.Empty:
                continue

            # Check resource usage
            if self.check_resource_limits():
                self.log_message(
                    "Resource limits exceeded. Stopping translation.")
                self.stop_translation()
                break

    def identify_speaker(self, audio_data):
        # Extract features (e.g., MFCCs)
        _, _, Sxx = signal.spectrogram(
            audio_data, fs=RATE, nperseg=256, noverlap=128)
        features = np.mean(Sxx, axis=1)
        self.speaker_features.append(features)

        if len(self.speaker_features) > 10:  # Wait for some samples before clustering
            self.speaker_kmeans.fit(self.speaker_features)
            speaker = self.speaker_kmeans.predict(
                [features])[0] + 1  # Speaker numbers start from 1
        else:
            speaker = 1  # Default speaker until we have enough samples

        return speaker

    def update_display(self, text):
        self.text_area.insert("end", text)
        self.text_area.see("end")
        self.log_message(text.strip())

    def log_message(self, message):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        self.log_area.insert("end", log_entry)
        self.log_area.see("end")

    def update_elapsed_time(self):
        if self.start_time:
            elapsed = int(time.time() - self.start_time)
            elapsed_str = str(datetime.timedelta(seconds=elapsed))
            self.elapsed_time_label.configure(
                text=f"Elapsed Time: {elapsed_str}")
        if self.is_translating:
            self.after(1000, self.update_elapsed_time)

    def update_resource_usage(self):
        cpu_usage = psutil.cpu_percent()
        memory_usage = psutil.virtual_memory().percent

        self.cpu_label.configure(text=f"CPU Usage: {cpu_usage}%")
        self.memory_label.configure(text=f"Memory Usage: {memory_usage}%")

        self.cpu_data.append(cpu_usage)
        self.memory_data.append(memory_usage)
        self.cpu_data = self.cpu_data[-60:]
        self.memory_data = self.memory_data[-60:]

        self.ax1.clear()
        self.ax2.clear()
        self.ax1.plot(self.cpu_data, color='skyblue')
        self.ax2.plot(self.memory_data, color='skyblue')
        self.ax1.set_ylabel("CPU %")
        self.ax2.set_ylabel("Memory %")
        self.ax1.set_ylim(0, 100)
        self.ax2.set_ylim(0, 100)
        self.ax1.set_title("CPU Usage")
        self.ax2.set_title("Memory Usage")
        self.canvas.draw()

        self.total_cpu_cores_label.configure(
            text=f"Total CPU Cores: {psutil.cpu_count()}")
        self.total_memory_label.configure(
            text=f"Total Memory: {psutil.virtual_memory().total // (1024 ** 2)} MB")
        self.available_memory_label.configure(text=f"Available Memory: {
                                              psutil.virtual_memory().available // (1024 ** 2)} MB")
        self.disk_usage_label.configure(
            text=f"Disk Usage: {psutil.disk_usage('/').percent}%")
        net_io = psutil.net_io_counters()
        self.network_io_label.configure(text=f"Network IO: Sent {
                                        net_io.bytes_sent // (1024 ** 2)} MB, Received {net_io.bytes_recv // (1024 ** 2)} MB")

        if self.is_translating:
            self.after(1000, self.update_resource_usage)

    def check_resource_limits(self):
        cpu_usage = psutil.cpu_percent()
        memory_usage = psutil.virtual_memory().percent
        return cpu_usage > self.cpu_limit or memory_usage > self.memory_limit

    def show_toast(self, message):
        tkmb.showinfo("Notification", message)


if __name__ == "__main__":
    app = TranslatorApp()
    app.mainloop()
