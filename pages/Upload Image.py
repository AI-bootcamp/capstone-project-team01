import streamlit as st
import tempfile
from ultralytics import YOLO
import chess
import chess.svg
from helpers import *
from chess_functions import *
import pandas as pd
import base64

# Set Streamlit page configuration
st.set_page_config(page_title="Mid Chess Game Detection", page_icon="♟️")

# Load YOLO model
model = YOLO('weights/bestV8.pt')

# Initialize session state
def initialize_session_state():
    if 'conf_threshold' not in st.session_state:
        st.session_state.conf_threshold = 0.7
    if 'chessboard' not in st.session_state:
        st.session_state.chessboard = chess.Board()
    if 'black_pawn_positions' not in st.session_state:
        st.session_state.black_pawn_positions = []
    if 'white_pawn_positions' not in st.session_state:
        st.session_state.white_pawn_positions = []
    if 'previous_board_status' not in st.session_state:
        st.session_state.previous_board_status = None
    if 'selected_position' not in st.session_state:
        st.session_state.selected_position = None
    if 'piece_choice' not in st.session_state:
        st.session_state.piece_choice = None
    if 'image_processed' not in st.session_state:
        st.session_state.image_processed = []
    if 'pending_changes' not in st.session_state:
        st.session_state.pending_changes = False

initialize_session_state()

# Page title and placeholders
st.title("Chess Game with Detection Midway")
warning = st.empty()
result_announcement = st.empty()
col1, col2, col3 = st.columns(3)
with col1:
    det_out = st.empty()
with col2:
    image_out = st.empty()
with col3:
    board_svg_placeholder = st.empty()

# Update board display if it exists in session state
if 'chessboard' in st.session_state:
    board_svg_placeholder.markdown(update_board_display(st.session_state.chessboard), unsafe_allow_html=True)

# Helper function to process uploaded image
def process_image(image_path):
    results = model.predict(source=image_path, conf=st.session_state.conf_threshold)
    if results:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        predicted_classes = results[0].boxes.cls
        class_names = model.names
        predicted_class_names = [class_names[int(cls)] for cls in predicted_classes]
        
        detection_vis = results[0].plot()
        det_out.image(detection_vis, channels="BGR", use_container_width=True)
        
        st.session_state.previous_board_status = order_detections(boxes, predicted_class_names)
        st.session_state.chessboard = update_board_and_extract_pawns(st.session_state.previous_board_status)
        st.session_state.image_processed.append(image_path)  # Mark image as processed
        board_svg_placeholder.markdown(update_board_display(st.session_state.chessboard), unsafe_allow_html=True)

# Helper function to update board and extract pawn positions
def update_board_and_extract_pawns(board_status):
    board = chess.Board(None)
    st.session_state.black_pawn_positions.clear()
    st.session_state.white_pawn_positions.clear()

    for row in range(8):
        for col in range(8):
            piece_color = board_status[row][col]
            if piece_color != 'empty':
                piece_symbol = 'P' if piece_color == 'white' else 'p'
                square = chess.square(col, 7 - row)
                board.set_piece_at(square, chess.Piece.from_symbol(piece_symbol))
                
                # Check for pawn and update positions
                piece = board.piece_at(square)
                if piece and piece.piece_type == chess.PAWN:
                    if piece.color == chess.BLACK:
                        st.session_state.black_pawn_positions.append(to_chess_notation(row, col))
                    elif piece.color == chess.WHITE:
                        st.session_state.white_pawn_positions.append(to_chess_notation(row, col))
    return board

# Helper function to convert row/column to chess notation
def to_chess_notation(row, col):
    return f"{chr(col + 97)}{8 - row}"

# Function to handle pawn selection and piece placement
def handle_pawn_selection(pawn_color):
    positions = (st.session_state.black_pawn_positions 
                 if pawn_color == "Black Pawns" 
                 else st.session_state.white_pawn_positions)
    
    if positions:
        st.session_state.selected_position = st.selectbox(f"{pawn_color} Positions:", options=positions)
        st.session_state.piece_choice = st.radio("Choose a piece to place:", available_pieces, index=0)

        if st.session_state.selected_position and st.session_state.piece_choice:
            st.session_state.pending_changes = True
            st.write("Click 'Apply Changes' to update the board.")
    else:
        st.write("Upload an image to detect pieces.")

# Apply changes to the chessboard
def apply_changes():
    if st.session_state.pending_changes:
        piece_symbol = piece_map[st.session_state.piece_choice]
        piece = chess.Piece.from_symbol(piece_symbol.lower() if st.session_state.pawn_choice == "Black Pawns" else piece_symbol)
        square = chess.parse_square(st.session_state.selected_position)
        st.session_state.chessboard.set_piece_at(square, piece)

        # Clear pending changes
        st.session_state.pending_changes = False

        # Update the displayed board
        board_svg_placeholder.markdown(update_board_display(st.session_state.chessboard), unsafe_allow_html=True)

# Upload and process image
uploaded_file = st.file_uploader("Upload a chess image", type=["jpg", "jpeg", "png", "bmp"])
if uploaded_file and not st.session_state.image_processed:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_image:
        temp_image.write(uploaded_file.read())
        temp_image_path = temp_image.name
        image_out.image(temp_image_path, caption="Uploaded Image", use_container_width=True)
        if not temp_image_path in st.session_state.image_processed:
            process_image(temp_image_path)

# Define available pieces and map
available_pieces = ["Knight", "Rook", "Bishop", "Queen", "King"]
piece_map = {
    "Knight": "N",
    "Rook": "R",
    "Bishop": "B",
    "Queen": "Q",
    "King": "K"
}

# Handle pawn selection based on user input
pawn_choice = st.radio("Select Pawn Color", ["White Pawns", "Black Pawns"], index=0)
st.session_state.pawn_choice = pawn_choice
handle_pawn_selection(pawn_choice)

# Apply changes button
if st.button("Apply Changes"):
    apply_changes()
