import sounddevice as sd
import numpy as np
import threading
import queue
import torch
import keyboard
from scipy.io import wavfile
from faster_whisper import WhisperModel
from transformers import pipeline
from gtts import gTTS
import pygame
import tempfile
import os
from tenacity import retry, stop_after_attempt, wait_exponential

from scipy import signal
import webrtcvad
import collections

class AudioProcessor:
    def __init__(self, sample_rate=16000, buffer_duration=30):
        self.sample_rate = sample_rate
        self.buffer_duration = buffer_duration  # seconds
        self.audio_queue = queue.Queue()
        self.is_recording = False
        self.vad = webrtcvad.Vad(3)  # Aggressive VAD
        self.buffer = collections.deque(maxlen=int(sample_rate * buffer_duration))
        self.min_audio_level = 0.01
        
    def audio_callback(self, indata, frames, time, status):
        if status:
            print(f"Status: {status}")
            
        # Convert to float32 and normalize
        audio_chunk = indata.flatten().astype(np.float32) / np.iinfo(np.int16).max
        
        # Check audio level
        audio_level = np.abs(audio_chunk).mean()
        if audio_level > self.min_audio_level:
            # Apply noise reduction
            filtered_audio = self._reduce_noise(audio_chunk)
            self.audio_queue.put(filtered_audio)
            self.buffer.extend(filtered_audio)
            
    def _reduce_noise(self, audio_chunk):
        # Simple noise reduction using bandpass filter
        nyquist = self.sample_rate / 2
        low = 300 / nyquist
        high = 3000 / nyquist
        b, a = signal.butter(4, [low, high], btype='band')
        return signal.filtfilt(b, a, audio_chunk)
        
    def start_recording(self):
        self.is_recording = True
        self.buffer.clear()
        self.stream = sd.InputStream(
            callback=self.audio_callback,
            channels=1,
            samplerate=self.sample_rate,
            blocksize=int(self.sample_rate * 0.1),  # 100ms chunks
            dtype=np.float32
        )
        self.stream.start()
        
    def stop_recording(self):
        if self.is_recording:
            self.is_recording = False
            self.stream.stop()
            self.stream.close()
            
    def get_audio_data(self):
        # Convert buffer to numpy array
        audio_data = np.array(list(self.buffer))
        
        # Apply VAD to trim silence
        if len(audio_data) > 0:
            audio_data = self._trim_silence(audio_data)
            
        # Ensure minimum duration (0.5 seconds)
        if len(audio_data) < self.sample_rate * 0.5:
            return np.array([])
            
        return audio_data
        
    def _trim_silence(self, audio_data):
        frame_duration = 30  # ms
        frame_size = int(self.sample_rate * frame_duration / 1000)
        
        # Convert float32 to int16 for VAD
        audio_int16 = (audio_data * 32768).astype(np.int16)
        
        # Find voice segments
        voiced_segments = []
        for i in range(0, len(audio_int16) - frame_size, frame_size):
            frame = audio_int16[i:i + frame_size]
            if self.vad.is_speech(frame.tobytes(), self.sample_rate):
                voiced_segments.append(audio_data[i:i + frame_size])
                
        if voiced_segments:
            return np.concatenate(voiced_segments)
        return np.array([])

class MedicalChatbot:
    def __init__(self):
        self.whisper = WhisperModel("large-v3", device="cuda" if torch.cuda.is_available() else "cpu")
        self.medical_qa = pipeline("question-answering", model="deepset/roberta-base-squad2")
        pygame.mixer.init()
        
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def transcribe_audio(self, audio_data, sample_rate):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
            wavfile.write(temp_wav.name, sample_rate, audio_data)
            segments, _ = self.whisper.transcribe(temp_wav.name, beam_size=5)
            text = " ".join([segment.text for segment in segments])
        os.unlink(temp_wav.name)
        return text
    
    def get_medical_response(self, query):
        context = """I am a medical AI assistant. I can help answer general medical questions 
        but cannot provide specific medical advice or diagnosis. Please consult a healthcare 
        professional for specific medical concerns."""
        
        result = self.medical_qa(question=query, context=context)
        return result['answer']
    
    def text_to_speech(self, text):
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_mp3:
            tts = gTTS(text=text, lang='en')
            tts.save(temp_mp3.name)
            pygame.mixer.music.load(temp_mp3.name)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
        os.unlink(temp_mp3.name)

def main():
    audio_processor = AudioProcessor()
    chatbot = MedicalChatbot()
    
    print("Press and hold SPACE to speak, release to get response...")
    
    while True:
        if keyboard.is_pressed('space'):
            if not audio_processor.is_recording:
                audio_processor.start_recording()
                print("Listening...")
        elif audio_processor.is_recording:
            audio_processor.stop_recording()
            print("Processing...")
            
            audio_data = audio_processor.get_audio_data()
            if len(audio_data) > 0:
                try:
                    # Transcribe speech to text
                    text = chatbot.transcribe_audio(audio_data, audio_processor.sample_rate)
                    print(f"You said: {text}")
                    
                    # Get medical response
                    response = chatbot.get_medical_response(text)
                    print(f"Response: {response}")
                    
                    # Convert response to speech
                    chatbot.text_to_speech(response)
                    
                except Exception as e:
                    print(f"Error: {str(e)}")
        
        if keyboard.is_pressed('esc'):
            break

if __name__ == "__main__":
    main()