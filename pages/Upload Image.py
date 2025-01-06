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


black_pawn_positions = []
white_pawn_positions = []

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

# board_svg_placeholder.markdown(update_board_display(board), unsafe_allow_html=True)

# Process the image to detect chess pieces
def process_image(imagePath):
    if not 'chessboard' in st.session_state:
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
            st.session_state.chessboard = update_board_and_extract_pawns(st.session_state.previous_board_status)
            print(st.session_state.chessboard)
    board_svg_placeholder.markdown(update_board_display(st.session_state.chessboard), unsafe_allow_html=True)

# Helper function to update the chessboard based on detected pieces
def update_board_and_extract_pawns(board_status):
    """
    Updates the chess.Board object based on the detected pieces' positions 
    and extracts the positions of pawns.
    """
    board = chess.Board(None)

    # Loop through the detected board status and update chessboard
    for row in range(8):
        for col in range(8):
            piece_color = board_status[row][col]  

            if piece_color != 'empty':
                piece_symbol = 'P' if piece_color == 'white' else 'p'  
                square = chess.square(col, 7 - row)  

                board.set_piece_at(square, chess.Piece.from_symbol(piece_symbol))
                
                # Check if the placed piece is a pawn and extract its position
                piece = board.piece_at(square)
                if piece and piece.piece_type == chess.PAWN:
                    if piece.color == chess.BLACK:
                        black_pawn_positions.append(to_chess_notation(row, col))
                    elif piece.color == chess.WHITE:
                        white_pawn_positions.append(to_chess_notation(row, col))

    return board

# Helper function to convert row/column to chess notation
def to_chess_notation(row, col):
    return f"{chr(col + 97)}{8 - row}"

# Upload and process image
uploaded_file = st.file_uploader("Upload a chess image", type=["jpg", "jpeg", "png", "bmp"])
if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_image:
        temp_image.write(uploaded_file.read())
        temp_image_path = temp_image.name

        image_out.image(temp_image_path, caption="Uploaded Image", use_container_width=True)

        process_image(temp_image_path)


pawn_choice = st.radio("Select Pawn Color", ("Black Pawns", "White Pawns"),None)
available_pieces = ["Knight", "Rook", "Bishop", "Queen", "King"]
piece_map = {
    "Knight": "N",  # Knight -> 'N'
    "Rook": "R",    # Rook -> 'R'
    "Bishop": "B",  # Bishop -> 'B'
    "Queen": "Q",   # Queen -> 'Q'
    "King": "K"     # King -> 'K'
}

if pawn_choice == "Black Pawns":
    if black_pawn_positions:  
        st.session_state.selected_position = st.selectbox("Black Pawn Positions:", options=black_pawn_positions)
        st.session_state.piece_choice = st.radio("Choose a piece to place:", available_pieces, None)

        if st.session_statepiece_choice:
            piece_symbol = piece_map.get(st.session_state.piece_choice)  
            piece = chess.Piece.from_symbol(piece_symbol.lower())  
            square = chess.parse_square(st.session_state.selected_position)  

            st.session_state.chessboard.set_piece_at(square, piece)
            st.session_state.chessboard = st.session_state.chessboard  

            # Update the displayed board
            board_svg_placeholder.markdown(update_board_display(st.session_state.chessboard), unsafe_allow_html=True)

    else:
        st.write("Upload an image to detect pieces.")
else:
    if white_pawn_positions: 
        st.session_state.selected_position = st.selectbox("White Pawn Positions:", options=white_pawn_positions)
        st.session_state.piece_choice = st.radio("Choose a piece to place:", available_pieces, None)

        if st.session_state.piece_choice:
            piece_symbol = piece_map.get(st.session_state.piece_choice)  
            piece = chess.Piece.from_symbol(piece_symbol) 
            square = chess.parse_square(st.session_state.selected_position)

            st.session_state.chessboard.set_piece_at(square, piece)
            st.session_state.chessboard = st.session_state.chessboard  

            # Update the displayed board
            board_svg_placeholder.markdown(update_board_display(st.session_state.chessboard), unsafe_allow_html=True)

    else:
        st.write("Upload an image to detect pieces.")
