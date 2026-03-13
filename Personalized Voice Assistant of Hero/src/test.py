from pydub import AudioSegment

# সরাসরি WAV ফাইল লোড করো
sound = AudioSegment.from_wav("temp_audio.wav")

# উদাহরণ: duration, frame rate check
print(f"Duration: {len(sound)/1000} seconds")
print(f"Frame rate: {sound.frame_rate}")