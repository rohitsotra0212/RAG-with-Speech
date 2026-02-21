import streamlit as st
import sounddevice as sd
import wave
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from openai import OpenAI

load_dotenv()

# -----------------------
# Setup API Clients
# -----------------------
openai_api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=openai_api_key)

llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=openai_api_key,
    temperature=0
)

# -----------------------
# Record Audio
# -----------------------
def record_audio(filename="command.wav", duration=5, samplerate=16000):
    audio = sd.rec(
        int(duration * samplerate),
        samplerate=samplerate,
        channels=1,
        dtype='int16'
    )
    sd.wait()

    with wave.open(filename, 'wb') as wave_file:
        wave_file.setnchannels(1)
        wave_file.setsampwidth(2)
        wave_file.setframerate(samplerate)
        wave_file.writeframes(audio.tobytes())

    return filename


# -----------------------
# Transcribe Audio
# -----------------------
def transcribe_audio(filename):
    with open(filename, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
        )

    return transcript.text


# -----------------------
# Generate 2-Line Answer
# -----------------------
def generate_answer(command_text):
    response = llm.invoke([
        SystemMessage(content="Answer in maximum 2 short lines only."),
        HumanMessage(content=command_text)
    ])

    return response.content


# -----------------------
# Text-to-Speech (NEW)
# -----------------------
def text_to_speech(text, output_file="response.mp3"):
    with client.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",
        voice="alloy",
        input=text
    ) as response:
        response.stream_to_file(output_file)

    return output_file


# -----------------------
# Streamlit UI
# -----------------------
st.set_page_config(page_title="Voice AI Assistant", page_icon="🎤")

st.title("🎤 Voice AI Assistant")
st.write("Record your voice → Get AI response → Hear it spoken")

duration = st.slider("Recording Duration (seconds)", 3, 10, 5)

if st.button("🎙 Record & Ask AI"):

    with st.spinner("Recording... Speak now!"):
        audio_file = record_audio(duration=duration)

    st.success("Recording complete!")

    with st.spinner("Transcribing audio..."):
        text = transcribe_audio(audio_file)

    st.subheader("📝 You said:")
    st.write(text)

    with st.spinner("Generating AI response..."):
        answer = generate_answer(text)

    st.subheader("🤖 AI Answer:")
    st.write(answer)

    with st.spinner("Speaking answer..."):
        audio_response = text_to_speech(answer)

    st.audio(audio_response)
