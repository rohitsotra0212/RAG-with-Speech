import sounddevice as sd
import numpy as np
import wave
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

import os
from dotenv import load_dotenv

load_dotenv()

## LangChain LLM (for chat/Generate answer)
llm = ChatOpenAI(model='gpt-4o-mini', api_key=os.getenv("OPENAI_API_KEY"), temperature=0)

## OpenAI client (for Whisper transcription)
client = OpenAI(api_key= os.getenv("OPENAI_API_KEY"))


## Step 1: Record Voice from Microphone
def record_audio(filename="command.wav", duration=5, samplerate=16000):
    print("Recording.... Speak Now!")
    audio = sd.rec(int(duration * samplerate), samplerate= samplerate, channels=1, dtype='int16')
    sd.wait()
    wave_file = wave.open(filename, 'wb')
    wave_file.setnchannels(1)
    wave_file.setsampwidth(2)
    wave_file.setframerate(samplerate)
    wave_file.writeframes(audio.tobytes())
    wave_file.close()
    print(f"Recoding saved at: {filename}")

## Step 2: Transcribe audio with Whisper
def transcibe_audio(filename="command.wav"):
    audio_file = open(filename, 'rb')
    transcript = client.audio.transcriptions.create(model="whisper-1",file= audio_file)

    command_text = transcript.text
    print(f"You said.. {command_text}")

    return command_text

## Step 3: Generate answer with GPT
def generate_answer(command_text):
    
    response = llm.invoke([
            SystemMessage(content="Answer in maximum 2 short lines only."),
            HumanMessage(content=command_text)
            ])

    answer_text = response.content
    print(f"AI Answer:\n{answer_text}")
    return answer_text

## Step 4: Convert answer to Speech
def speech_answer(answer_text, filename="answer.mp3"):
    speech_file = open(filename, 'wb')
    tts_response = client.audio.speech.create(model='gpt-4o-mini-tts',
                                              voice='alloy',
                                              input=answer_text
                                              )

    speech_file.write(tts_response.read())
    speech_file.close()
    print("Thank You!!")

if __name__ == "__main__":
    record_audio()
    command_text = transcibe_audio()
    answer_text = generate_answer(command_text)
    speech_answer(answer_text)
