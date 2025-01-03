import streamlit as st
import cv2
import tempfile
from ultralytics import YOLO
from helpers import *
import chess
import chess.svg
from io import BytesIO
import base64
from reportlab.pdfgen import canvas
import pandas as pd

# Load YOLO model
model = YOLO('weights/bestV5.pt')

# Initialize board and move history
if 'board' not in st.session_state:
    st.session_state.board = chess.Board()
    st.session_state.move_history = []
    st.session_state.chessboard = [
        ['br', 'bn', 'bb', 'bq', 'bk', 'bb', 'bn', 'br'],
        ['bp', 'bp', 'bp', 'bp', 'bp', 'bp', 'bp', 'bp'],
        ['.', '.', '.', '.', '.', '.', '.', '.'],
        ['.', '.', '.', '.', '.', '.', '.', '.'],
        ['.', '.', '.', '.', '.', '.', '.', '.'],
        ['.', '.', '.', '.', '.', '.', '.', '.'],
        ['wp', 'wp', 'wp', 'wp', 'wp', 'wp', 'wp', 'wp'],
        ['wr', 'wn', 'wb', 'wq', 'wk', 'wb', 'wn', 'wr']
    ]
    st.session_state.previous_board_status = [
        ['black', 'black', 'black', 'black', 'black', 'black', 'black', 'black'],
        ['black', 'black', 'black', 'black', 'black', 'black', 'black', 'black'],
        ['empty', 'empty', 'empty', 'empty', 'empty', 'empty', 'empty', 'empty'],
        ['empty', 'empty', 'empty', 'empty', 'empty', 'empty', 'empty', 'empty'],
        ['empty', 'empty', 'empty', 'empty', 'empty', 'empty', 'empty', 'empty'],
        ['empty', 'empty', 'empty', 'empty', 'empty', 'empty', 'empty', 'empty'],
        ['white', 'white', 'white', 'white', 'white', 'white', 'white', 'white'],
        ['white', 'white', 'white', 'white', 'white', 'white', 'white', 'white']
    ]
    st.session_state.move_table = pd.DataFrame(columns=["Piece", "From", "To"])

board = st.session_state.board
move_history = st.session_state.move_history
move_table = st.session_state.move_table

st.title("Chess Game with Detection")

# Sidebar settings
st.sidebar.title("Settings")
conf_threshold = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.05, 0.05)
grid_rows = grid_cols = 8

# Function to render chessboard as SVG
def render_board(board):
    return chess.svg.board(board=board)

# Display board
board_svg_placeholder = st.empty()
def update_board_display():
    board_svg = render_board(board)
    encoded_svg = base64.b64encode(board_svg.encode('utf-8')).decode('utf-8')
    board_svg_placeholder.markdown(f'<img src="data:image/svg+xml;base64,{encoded_svg}" width="400"/>', unsafe_allow_html=True)

update_board_display()

# Detect move from board status
def detect_move(previous_board_status, new_board_status, chessboard):
    move = {}
    for row in range(len(previous_board_status)):
        for col in range(len(previous_board_status[row])):
            if previous_board_status[row][col] != new_board_status[row][col]:
                if new_board_status[row][col] == 'empty' and previous_board_status[row][col] != 'empty':
                    move['start'] = (row, col)
                    move['piece'] = chessboard[row][col]
                elif previous_board_status[row][col] == 'empty' and new_board_status[row][col] != 'empty':
                    move['end'] = (row, col)
                elif previous_board_status[row][col] == 'black' and new_board_status[row][col] == 'white':
                    move['end'] = (row, col)
                    move['eliminated'] = chessboard[row][col]
                elif previous_board_status[row][col] == 'white' and new_board_status[row][col] == 'black':
                    move['end'] = (row, col)
                    move['eliminated'] = chessboard[row][col]
    return move

# Update chessboard after detecting move
def update_chessboard(move, chessboard):
    start = move['start']
    end = move['end']
    piece = move['piece']
    chessboard[start[0]][start[1]] = '.'
    chessboard[end[0]][end[1]] = piece
    return chessboard

# Process the image to detect chess pieces
def process_image(imagePath):
    spaces = create_grid(grid_rows, grid_cols)
    results = model.predict(source=imagePath, conf=conf_threshold)
    img = cv2.imread(imagePath)

    if results:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        predicted_classes = results[0].boxes.cls
        class_names = model.names
        predicted_class_names = [class_names[int(cls)] for cls in predicted_classes]

        new_shape, lm, bm = get_new_shape(boxes, img.shape[:2])  # Ensure lm and bm are calculated properly

        # Debugging to ensure lm and bm are calculated
        st.write(f"lm: {lm}, bm: {bm}")

        occupancy = map_detections_to_spaces(boxes, spaces, predicted_class_names, new_shape, grid_rows, grid_cols, lm, bm)
        st.write("Occupancy data:", occupancy)
        new_board_status = map_occupancy_to_board_status(occupancy)
        st.write("New board status:", new_board_status)

        move = detect_move(st.session_state.previous_board_status, new_board_status, st.session_state.chessboard)
        st.session_state.previous_board_status = new_board_status

        if 'start' in move and 'end' in move:
            start_square = f"{chr(97 + move['start'][1])}{8 - move['start'][0]}"
            end_square = f"{chr(97 + move['end'][1])}{8 - move['end'][0]}"
            chess_move = chess.Move.from_uci(f"{start_square}{end_square}")

            if chess_move in board.legal_moves:
                board.push(chess_move)
                move_history.append(f"{move['piece']} from {start_square} to {end_square}")
                move_table.loc[len(move_table)] = [move['piece'], start_square, end_square]
                st.session_state.chessboard = update_chessboard(move, st.session_state.chessboard)
                update_board_display()

# Upload and process image
uploaded_file = st.file_uploader("Upload a chess image", type=["jpg", "jpeg", "png", "bmp"])
if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_image:
        temp_image.write(uploaded_file.read())
        temp_image_path = temp_image.name

        st.image(temp_image_path, caption="Uploaded Image", width=700)

        if st.button("Process Image"):
            process_image(temp_image_path)

# Display move table
st.write("### Move History Table")
st.dataframe(move_table)
