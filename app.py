import streamlit as st
from modules.stt import transcribe_audio_whisper
from modules.vision import encode_image_clip
from modules.diagnosis import get_doctor_response
from modules.tts import speak_with_gtts
import tempfile

st.set_page_config(page_title="AI Doctor", page_icon="🩺", layout="centered")

st.title("🩺 AI Doctor - Voice & Vision Assistant")
st.markdown("Ask your question by **voice or typing**, and (optionally) upload a skin-related image.")

# Use tabs for cleaner input
tab1, tab2 = st.tabs(["🎤 Voice or Text", "🖼️ Upload Image"])

query_text = ""
image_features = None

# --- 🎤 Tab 1: Voice or Text ---
with tab1:
    st.subheader("Voice or Text Input")

    audio_file = st.file_uploader("🎙️ Upload voice (mp3/wav)", type=["mp3", "wav"])
    manual_input = st.text_input("Or type your symptoms")

    if audio_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_file.read())
            audio_path = tmp.name
        st.info("Transcribing audio...")
        query_text = transcribe_audio_whisper(audio_path)
        st.success("Transcription: " + query_text)

    elif manual_input:
        query_text = manual_input

# --- 🖼️ Tab 2: Image Upload ---
with tab2:
    st.subheader("Upload Skin Image (Optional)")
    image_file = st.file_uploader("📸 Upload image", type=["jpg", "jpeg", "png", "webp"])

    if image_file:
        st.image(image_file, caption="Uploaded Image", use_column_width=True)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as img_tmp:
            img_tmp.write(image_file.read())
            image_path = img_tmp.name
        image_features = encode_image_clip(image_path)

# --- 🧠 Submit Button ---
if st.button("🧠 Get Doctor's Advice"):
    if not query_text:
        st.warning("❗ Please provide voice or text input first.")
    else:
        with st.spinner("🩺 Analyzing..."):
            response = get_doctor_response(query=query_text, encoded_image=image_features)
        st.subheader("🧠 Doctor Says:")
        st.success(response)

        # Doctor's voice
        audio_path = speak_with_gtts(response)
        st.audio(audio_path, format="audio/mp3")
