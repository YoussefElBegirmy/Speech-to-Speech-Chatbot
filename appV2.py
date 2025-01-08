import torch
from faster_whisper import WhisperModel
import pyaudio
import wave
import msvcrt  # for Windows
from groq import Groq
from pydub import AudioSegment, effects
import os
from datetime import datetime
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter,RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings,HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFaceHub
from langchain_community.document_loaders import WebBaseLoader
import bs4
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA

load_dotenv()

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
        self.save_recorded_file_to_disk()
        print("Recording stopped")

    def save_recorded_file_to_disk(self, filename="output.wav"):
        sound_file = wave.open(filename, "wb")
        sound_file.setnchannels(1)
        sound_file.setsampwidth(self.audio.get_sample_size(pyaudio.paInt16))
        sound_file.setframerate(44100)
        sound_file.writeframes(b''.join(self.frames))
        sound_file.close()

    def save_recorded_file_with_timestamp(self):
        
        # Create directory if it doesn't exist
        folder_path = "Recored_audio"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        # Generate timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(folder_path, f"recording_{timestamp}.wav")
        
        # Save the audio file
        self.save_recorded_file_to_disk(filename=filename)


class Speech_to_text:
    def __init__(self,model_size):
        self.model_size = model_size
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    def transcribe_audio(self, audio_file="output.wav"):
        with open(audio_file, "rb") as audio_file:
            self.Transcript = self.client.audio.transcriptions.create(
                model="whisper-large-v3",
                file=audio_file,
                language="en"
            )
        print(self.Transcript.text)
        with open('output.txt', 'a', encoding='utf-8') as f:
            f.write("\n-----------------------------------------------------------------\n")
            f.write(self.Transcript.text)
    def save_Transcript_file_with_timestamp(self):
        
        # Create directory if it doesn't exist
        folder_path = "Transcribed_text"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        # Generate timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(folder_path, f"Transcript_{timestamp}.txt")
        
        # Save the Transcript file
        with open(filename, 'a', encoding='utf-8') as f:
            f.write("\n-----------------------------------------------------------------\n")
            f.write(self.Transcript.text)

class chatbot:


if __name__ == "__main__":
    audio = Audio()
    model_size = "large-v3"
    speech = Speech_to_text(model_size)
    print("Press 'q' to start recording and transcribing, or 'e' to exit")
    while True:
        if msvcrt.kbhit():
            key = msvcrt.getch().decode().lower()
            if key == 'q':
                audio.record_audio()
                speech.transcribe_audio()
                audio.save_recorded_file_with_timestamp()
                speech.save_Transcript_file_with_timestamp()
                print("Press 'q' to start recording and transcribing, or 'e' to exit")
            elif key == 'e':
                print("Exiting program...")
                break