import streamlit as st
from modules.stt import transcribe_audio_whisper
from modules.vision import encode_image_clip
from modules.diagnosis import get_doctor_response
from modules.tts import speak_with_gtts
import os
import tempfile

st.set_page_config(page_title="AI Doctor", page_icon="🩺")

st.title("🩺 AI Doctor - Voice + Vision")
st.markdown("Describe your issue or upload your voice. Also, upload an image of the affected skin area (optional).")

# Voice input
audio_file = st.file_uploader("🎤 Upload your voice file (.mp3, .wav)", type=["mp3", "wav"])
query_text = ""

if audio_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_file.read())
        tmp_path = tmp.name
    st.success("Transcribing...")
    query_text = transcribe_audio_whisper(tmp_path)
    st.write("**Transcribed Query:**", query_text)

# Manual input
manual_input = st.text_input("📝 Or type your symptoms manually:")

if manual_input:
    query_text = manual_input

# Image input
image_file = st.file_uploader("🖼️ Upload an image of your skin/issue (optional)", type=["jpg", "jpeg", "png", "webp"])
image_features = None

if image_file:
    st.image(image_file, caption="Uploaded Image", use_column_width=True)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as img_tmp:
        img_tmp.write(image_file.read())
        image_path = img_tmp.name
    image_features = encode_image_clip(image_path)

# Diagnosis
if st.button("🧠 Get AI Doctor Response") and query_text:
    st.info("Analyzing...")
    response = get_doctor_response(query=query_text, encoded_image=image_features)
    st.markdown("**🩺 Doctor says:**")
    st.success(response)

    # TTS output
    st.audio(speak_with_gtts(response), format="audio/mp3")
elif st.button("🧠 Get AI Doctor Response") and not query_text:
    st.warning("Please provide voice or text input first.")
