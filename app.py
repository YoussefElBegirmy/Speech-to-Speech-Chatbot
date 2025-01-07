import torch
from faster_whisper import WhisperModel
import pyaudio
import wave
import msvcrt  # for Windows

if torch.cuda.is_available():
    print("CUDA is available. You can use GPU for training.")
    print(f"Number of GPUs available: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("CUDA is not available. You cannot use GPU for training.")

class Audio:
    def record_audio(self):
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True, frames_per_buffer=1024)
        self.frames = []
        print("Recording... Press 'q' to stop")
        try:
            while True:
                if msvcrt.kbhit() and msvcrt.getch().decode().lower() == 'q':
                    break
                data = self.stream.read(1024)
                self.frames.append(data)
        except Exception as e:
            print(f"Error recording: {str(e)}")
        
        self.stream.stop_stream()
        self.stream.close()
        self.audio.terminate()
        
        sound_file = wave.open("output.wav", "wb")
        sound_file.setnchannels(1)
        sound_file.setsampwidth(self.audio.get_sample_size(pyaudio.paInt16))
        sound_file.setframerate(44100)
        sound_file.writeframes(b''.join(self.frames))
        sound_file.close()
        print("Recording stopped")


class Speech_to_text:
    def __init__(self,model_size):
        self.model_size = model_size
        self.model = WhisperModel(self.model_size, device="cuda" if torch.cuda.is_available() else "cpu")
    def transcribe_audio(self):
        self.audio.record_audio()
        segments, info = self.model.transcribe("output.wav", beam_size=5)
        print("Detected language '%s' with probability %f" % (info.language, info.language_probability))
        with open('output.txt', 'a', encoding='utf-8') as f:
            f.write("-----------------------------------------------------------------\n")
        for segment in segments:
            print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
            # Write transcribed text to file
            with open('output.txt', 'a', encoding='utf-8') as f:
                f.write("[%.2fs -> %.2fs] %s\n" % (segment.start, segment.end, segment.text))

if __name__ == "__main__":
    Audio1 = Audio()
    speech = Speech_to_text()
    model_size = "large-v3"
    print("Press 'q' to start recording and transcribing, or 'e' to exit")
    while True:
        if msvcrt.kbhit():
            key = msvcrt.getch().decode().lower()
            if key == 'q':
                Audio1.record_audio()
                speech.transcribe_audio()
                print("Press 'q' to start recording and transcribing, or 'e' to exit")
            elif key == 'e':
                print("Exiting program...")
                break