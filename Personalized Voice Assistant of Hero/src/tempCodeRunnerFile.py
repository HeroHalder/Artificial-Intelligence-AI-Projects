# src/run_assistant.py
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
from infer import predict
import time
import os
import webbrowser
from datetime import datetime

DURATION = 3  # seconds
SAMPLE_RATE = 16000  # Hz

# Map labels to commands
COMMANDS = {
    0: 'play_music',
    1: 'set_alarm',
    2: 'tell_time',
    3: 'stop_music',
}

MUSIC_PATH = r"D:\Personalized Voice Assistant of Hero\music.mp3"  # music file path

def record_audio(duration=DURATION, sample_rate=SAMPLE_RATE):
    print(f"\nRecording for {duration} seconds...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
    sd.wait()
    audio = (audio * 32767).astype(np.int16)
    wav.write("temp_audio.wav", sample_rate, audio)
    print("Recording saved as temp_audio.wav")
    return "temp_audio.wav"

def play_music():
    if os.path.exists(MUSIC_PATH):
        print("🎵 Playing music...")
        os.startfile(MUSIC_PATH)  # Windows-only
    else:
        print("❌ Music file not found!")

def stop_music():
    # This is tricky in Windows console without extra libraries
    print("🛑 Stop music command received (manual stop required)")

def set_alarm(seconds=5):
    print(f"⏰ Alarm set for {seconds} seconds from now...")
    time.sleep(seconds)
    print("🔔 Alarm ringing!")
    # Optional: play a sound file as alarm
    # os.startfile(r"D:\Personalized Voice Assistant of Hero\alarm.mp3")

def tell_time():
    now = datetime.now().strftime("%H:%M:%S")
    print(f"🕒 Current time: {now}")

def execute_command(label):
    command = COMMANDS.get(label, 'unknown')
    print(f"Executing command: {command}")
    if command == 'play_music':
        play_music()
    elif command == 'stop_music':
        stop_music()
    elif command == 'set_alarm':
        set_alarm(5)  # default 5 sec for testing
    elif command == 'tell_time':
        tell_time()
    else:
        print("❌ Unknown command.")

def main():
    model_path = 'models/baseline.h5'  # or fine-tuned model
    print("=== Personalized Voice Assistant Started ===")
    while True:
        cmd = input("\nPress Enter to record (or type 'exit' to quit): ")
        if cmd.lower() == 'exit':
            print("Exiting assistant. Goodbye! 👋")
            break
        audio_path = record_audio()
        label, prob = predict(audio_path, model_path=model_path)
        print(f"Predicted Command: {COMMANDS.get(label,'unknown')}, Probability: {prob:.2f}")
        execute_command(label)

if __name__ == "__main__":
    main()
