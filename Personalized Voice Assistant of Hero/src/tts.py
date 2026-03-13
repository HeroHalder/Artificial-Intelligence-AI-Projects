# TTS function
def speak(text):
    print("Assistant:", text)
    lang = 'bn' if any('\u0980' <= c <= '\u09FF' for c in text) else 'en'

    # Temporary file
    temp_path = os.path.join(tempfile.gettempdir(), "temp_tts.mp3")
    tts = gTTS(text=text, lang=lang)
    tts.save(temp_path)

    try:
        # Remove quotes, use raw path
        playsound(temp_path)
    except Exception as e:
        print("Error playing sound:", e)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
