from fastapi import File
import streamlit as st
from streamlit_option_menu import option_menu
import requests
from pydub import AudioSegment

with st.sidebar:
    selected = option_menu(
        menu_title="Main Menu",  # required
        options=["Home", "Project Design", "Meet The Team"],  # required
        icons=["house", "diagram-2", "people"],  # optional
        menu_icon="cast",  # optional
        default_index=0,  # optional
    )


if selected == "Home":
    st.markdown(
    """<h2 style='text-align: center; color: purple;font-size:60px;margin-top:-50px;'>Music Genre Classification</h2>""",
    unsafe_allow_html=True)
    global type
    UploadAudio = st.file_uploader("Upload Music To Classify", type=["wav", "mp3"])
    st.markdown(
        """<h3 style='color:purple;'> Play: </h3>""",
        unsafe_allow_html=True)
    st.audio(UploadAudio)

    if st.button("Predict"):
        if UploadAudio is not None:
            if type == "mp3":
                UploadAudio = AudioSegment.from_mp3(UploadAudio)
                UploadAudio.export("file.wav", format="wav")
            response = requests.post("http://127.0.0.1:8000/predict", data= UploadAudio)
            prediction =response
            st.success(f"You're Listening to: {prediction}")


if selected == "Project Design":
    st.markdown(
    """<h2 style='text-align: center; color: purple;font-size:60px;margin-top:-50px;'>Our Project Holistic View</h2>""",
    unsafe_allow_html=True)
if selected == "Meet The Team":
    st.markdown(
    """<h2 style='text-align: center; color: purple;font-size:60px;margin-top:-50px;'>Meet Our Amazing Team</h2>""",
    unsafe_allow_html=True)