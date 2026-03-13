# src/run_assistant_gtts.py

import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import speech_recognition as sr
from gtts import gTTS
from playsound import playsound
from src.infer import predict
import os, uuid

# ---------------- CONFIG ----------------
DURATION = 3.0  # seconds
MIC_INDEX = 1   # ✅ DirectSound backend microphone index
CONF_THRESHOLD = 0.75

# ---------------- TTS ----------------
def speak(text):
    print("Assistant:", text)
    try:
        lang = 'bn' if any('\u0980' <= c <= '\u09FF' for c in text) else 'en'
        mp3_path = f"tts_{uuid.uuid4().hex}.mp3"
        tts = gTTS(text=text, lang=lang)
        tts.save(mp3_path)
        playsound(mp3_path)
        os.remove(mp3_path)
    except Exception as e:
        print("Error in TTS:", e)

# ---------------- RECORD ----------------
def record_once(out="temp.wav"):
    device_info = sd.query_devices(MIC_INDEX, 'input')
    sample_rate = int(device_info['default_samplerate'])
    print(f"Recording {DURATION}s from device {MIC_INDEX} at {sample_rate} Hz...")

    audio = sd.rec(int(DURATION * sample_rate), samplerate=sample_rate, channels=1, device=MIC_INDEX)
    sd.wait()

    max_amp = np.max(np.abs(audio))
    if max_amp > 0:
        audio = audio / max_amp * 0.99

    audio = (audio * 32767).astype(np.int16)
    wav.write(out, sample_rate, audio)
    return out

# ---------------- COMMAND MAP ----------------
COMMAND_MAP = {
    # Bangla Numbers
    "শূন্য": "তুমি সংখ্যা শূন্য বলেছ।",
    "এক": "তুমি সংখ্যা এক বলেছ।",
    "দুই": "তুমি সংখ্যা দুই বলেছ।",
    "তিন": "তুমি সংখ্যা তিন বলেছ।",
    "ছয়": "তুমি সংখ্যা ছয় বলেছ।",

    # English Numbers
    "zero": "You said number zero.",
    "one": "You said number one.",
    "two": "You said number two.",
    "three": "You said number three.",
    "six": "You said number six.",

    # Bangla Greetings & Commands
    "হ্যালো": "হেই! তুমি কেমন আছো?",
    "কেমন আছো": "হিরো, আমি ভালো আছি। তুমি কেমন আছো?",
    "ধন্যবাদ": "স্বাগতম!",
    "হ্যাঁ": "ঠিক আছে, নিশ্চিত।",
    "না": "ঠিক আছে, বাতিল।",
    "বিদায়": "বিদায় হিরো! আবার দেখা হবে।",
    "চল": "চল শুরু করি! 🚀",
    "থাম": "এখন থামছি।",
    "হ্যালো হিরো": "Hello Hero! Nice to hear you.",

    # English Greetings & Commands
    "hello": "Hello Hero! Nice to hear you.",
    "how are you": "I am doing great. How about you?",
    "thanks": "You're welcome!",
    "thank you": "You're welcome!",
    "yes": "Okay, confirmed.",
    "no": "Alright, cancelled.",
    "goodbye": "Goodbye! See you again.",
    "go": "Let's start.",
    "stop": "Stopping now.",
    "play music": "Playing your favorite music now.",
    "set alarm": "Setting an alarm for you.",
    "weather": "The weather is sunny with a chance of rain later today.",
    "next": "Skipping to the next item.",
    "previous": "Going back to the previous item.",
    "creator": "This voice assistant was created by Hero.",
    "who are you": "I am your personal voice assistant.",
    "what can you do": "I can recognize commands and respond accordingly.",
    "tell me a joke": "Why did the scarecrow win an award? Because he was outstanding in his field!",
    "who is the creator of you": "Hero Halder is the creator of me.",
    "what is your name": "I am your personal voice assistant.",
}

# ---------------- PHONETIC / Banglish MAP ----------------
PHONETIC_MAP = {
    "গো": "go",
    "স্টপ": "stop",
    "গুডবাই": "goodbye",
    "থ্যাংকস": "thanks",
    "হাউ আর ইউ": "how are you",
    "হোয়াট ইজ ইওর নেম": "what is your name",
}

IGNORED_LABELS = ["sheila"]  # dataset bias ignore

# ---------------- TEXT MAPPING ----------------
def normalize_text(text):
    t = text.lower().strip()
    if t in PHONETIC_MAP:
        t = PHONETIC_MAP[t]
    return t

def map_from_text(text):
    if not text:
        return None
    t = normalize_text(text)
    if t in COMMAND_MAP:
        return COMMAND_MAP[t]
    for k in COMMAND_MAP:
        if k in t:
            return COMMAND_MAP[k]
    return None
# ---------------- STT ----------------
recognizer = sr.Recognizer()

def speech_to_text(wav_path):
    with sr.AudioFile(wav_path) as source:
        audio = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio, language="bn-BD")
        if text:
            return text
    except:
        pass
    try:
        text = recognizer.recognize_google(audio, language="en-US")
        return text
    except:
        return None

# ---------------- MAIN LOOP ----------------
if __name__ == "__main__":
    print("Assistant ready. Press Enter to speak (or type 'exit'). Voice will always reply.")

    try:
        while True:
            cmd = input()
            if cmd.lower() == "exit":
                break

            wav_path = record_once()

            # 1️⃣ STT mapping (priority)
            text = speech_to_text(wav_path)
            reply = None
            if text:
                print("STT:", text)
                reply = map_from_text(text)
                if reply:
                    speak(reply)
                    continue

            # 2️⃣ Dataset fallback (only if STT failed)
            label, prob = predict(wav_path)
            print(f"Dataset guess: {label} ({prob:.2f})")

            if label.lower() in IGNORED_LABELS:
                speak("Sorry, I could not understand. Please try again.")
            elif prob >= CONF_THRESHOLD:
                fallback_reply = COMMAND_MAP.get(label.lower())
                if fallback_reply:
                    speak(fallback_reply)
                else:
                    speak(f"I recognized '{label}', but no mapping reply is set.")
            else:
                speak("Sorry, I could not understand. Please try again.")

    except KeyboardInterrupt:
        print("\nAssistant stopped safely.")
