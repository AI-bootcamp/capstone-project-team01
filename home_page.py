import streamlit as st
from PIL import Image
import base64

# Page Configuration
st.set_page_config(
    page_title="Checkmate",
    page_icon="♟️",
    layout="wide",
)

# Convert Image to Base64
def get_base64_image(img_path):
    with open(img_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Get the base64 string of the uploaded image
base64_image = get_base64_image("Kingandqueenchesspieces-GettyImages-667749253-59f9021222fa3a0011f661f2.jpg")

# Custom CSS for Header Image
st.markdown(
    f"""
    <style>
    .header {{
        background-image: url("data:image/png;base64,{base64_image}");
        background-size: cover;
        height: 300px;
        text-align: center;
        color: white;
        font-size: 3rem;
        line-height: 300px;
        font-weight: bold;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# Header Section with Background Image
st.markdown(
    "<div class='header'>Checkmate Project ♟️</div>",
    unsafe_allow_html=True
)

# Title Section
st.markdown("### Track, Analyze, and Record Chess Games in Real Time")

# Project Introduction
st.write(
    "Welcome to the Checkmate! This project implements a real-time chess game move detection using YOLO (You Only Look Once), OpenCV and Streamlit. The application utilizes a webcam feed to analyze and recognize chessboard movements, tracks the game progress, and generates a move history and exports it to a PDF file. The application provides legal move suggestions to the user based on the current board state, ensuring players are aware of all possible actions. Users can also play against Stockfish, offering an interactive way to practice and test strategies against a high-level chess bot. Stockfish is exclusively used when the user opts to play against the bot, and not for suggesting moves during regular gameplay."
)

# Features Section
st.markdown("## Features")
features = [
    "Real-time Chess Detection: Use the webcam to capture and analyze chessboard movements.",
    "Game Recording: Record every move and save the full game in algebraic chess notation format.",
    "Move History Tracking: Logs each move made by white and black players.",
    "Play Against Stockfish: Users can play directly against Stockfish, simulating real gameplay and testing different strategies.",
    "Illegal Move Detection: Alerts if an illegal move is detected.",
    "Legal Move Suggestions: Show all possible legal moves for the player's pieces.",
    "Move Evaluation: Evaluates the quality of a move by comparing the board state before and after the move.",
    "Export to PDF: Download the move history as a PDF file."
]
for feature in features:
    st.write(f"- **{feature}**")

# Footer Section
st.markdown(
    "---\nDeveloped by Tariq Alshammari, Lamees Aloqlan, Sama Aldawayhie, Basel Felemabn"
)
