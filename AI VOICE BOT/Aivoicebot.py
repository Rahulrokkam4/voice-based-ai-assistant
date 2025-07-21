import os
import re
import time
import pyttsx3
import tempfile
from groq import Groq
from dotenv import load_dotenv
import speech_recognition as sr


load_dotenv()

client = Groq(api_key=os.getenv("GROQCLOUD_API_KEY"))


class AivoiceAssistant:
    def __init__(self, qa_chain):
        self.qa = qa_chain

    # voice engine
    def speak(self, text):
        try:
            print(f"Assistant: {text}")
            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            print("speech error :", e)

    # listen function
    def listen(self):
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            print("Listening...")
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source)
        try:
            # Save audio to temporary WAV file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
                temp_audio.write(audio.get_wav_data())
                temp_audio_path = temp_audio.name
            # Transcribe using Groq Whisper API
            with open(temp_audio_path, "rb") as file:
                transcription = client.audio.transcriptions.create(
                    file=file,
                    model="whisper-large-v3-turbo",
                    response_format="verbose_json",
                    language="en"
                )
            os.remove(temp_audio_path)  # Clean up temp file
            command = transcription.text
            print(f"You said: {command}")
            return command
        except Exception as e:
            print("Error during listening or transcription:", e)
            self.speak("Sorry, I couldn't understand.")
            return ""

    # give answer to user using RAG
    def ask_gpt(self, question):
        try:
            response = self.qa.invoke({"question": question})
            return response["result"]
        except Exception as e:
            print("GPT/RAG Error:", e)
            return "Sorry, I had a problem retrieving the answer."

    # dectect intent sending mail
    def detect_email_intent(self, command):
        keywords = ["send email", "mail", "contact", "forward", "message", "email", "meet", "meeting", "book appointment", "appointment"]
        return any(word in command.lower() for word in keywords)



    # extract email from rag usin namel̥
    def extract_email_from_name(self, response):
        response = self.qa.invoke({"question": f"What is the email of {response}"})
        email_text = response["result"]
        match = re.search(r'[\w\.-]+@[\w\.-]+',email_text)
        if match:
            return match.group().strip(" .;")
        else:
            return None

    # Break the conversation
    def is_goodbye(self, text):
        texts = ["that's all", "nothing else", "thank you", "thanks", "bye", "goodbye", "we're done"]
        return any(phrase in text.lower() for phrase in texts)

