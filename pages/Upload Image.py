import streamlit as st
import tempfile
from ultralytics import YOLO
import chess
import chess.svg
from helpers import *
from chess_functions import *
import pandas as pd
import base64


st.set_page_config(page_title="Mid Chess Game Detection", page_icon="♟️")

# Load YOLO model
model = YOLO('weights/bestV8.pt')


# Initialize board and move history
if 'board' not in st.session_state:
    st.session_state.board = chess.Board(None) 

# Confidence Threshold
st.session_state.conf_threshold = 0.7
board = st.session_state.board

# Placeholders
st.title("Chess Game with Detection midway")
warning = st.empty() 
result_announcement = st.empty()
col1, col2 , col3 = st.columns(3)
with col1:
    det_out = st.empty()
with col2:
    image_out = st.empty()
with col3:
    board_svg_placeholder = st.empty()

board_svg_placeholder.markdown(update_board_display(board), unsafe_allow_html=True)

# Process the image to detect chess pieces
def process_image(imagePath):
    results = model.predict(source=imagePath, conf=st.session_state.conf_threshold)

    if results:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        predicted_classes = results[0].boxes.cls
        class_names = model.names
        predicted_class_names = [class_names[int(cls)] for cls in predicted_classes]

        detection_vis = results[0].plot()
        det_out.image(detection_vis, channels="BGR", use_container_width=True)
        st.session_state.previous_board_status = order_detections(boxes, predicted_class_names)
        # Update the board based on detection results
        st.session_state.chessboard = update_chess_board_from_detection(st.session_state.previous_board_status)
        print(st.session_state.chessboard)
        board_svg_placeholder.markdown(update_board_display(st.session_state.chessboard), unsafe_allow_html=True)

# Helper function to update the chessboard based on detected pieces
def update_chess_board_from_detection(board_status):
    """
    Updates the chess.Board object based on the detected pieces' positions.
    """
    board = chess.Board(None)

    # Iterate through the board status and set pieces in the correct squares
    for row in range(8):
        for col in range(8):
            piece_color = board_status[row][col]
            if piece_color != 'empty':
                piece_symbol = 'P' if piece_color == 'white' else 'p'  
                square = chess.square(col, 7 - row) 
                board.set_piece_at(square, chess.Piece.from_symbol(piece_symbol))

    return board

# Upload and process image
uploaded_file = st.file_uploader("Upload a chess image", type=["jpg", "jpeg", "png", "bmp"])
if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_image:
        temp_image.write(uploaded_file.read())
        temp_image_path = temp_image.name

        image_out.image(temp_image_path, caption="Uploaded Image", use_container_width=True)

        process_image(temp_image_path)
