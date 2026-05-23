import requests
from utils.youtube_api import search_youtube_videos, get_youtube_api_key
import streamlit as st

st.secrets["YOUTUBE_API_KEY"] = "dummy"

# But wait, we want to test using the actual key... let me run the real function.
