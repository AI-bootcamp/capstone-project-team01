import streamlit as st
import cv2
from ultralytics import YOLO
from helpers import *
import chess
import chess.svg
from io import BytesIO
import base64
from reportlab.pdfgen import canvas
import pandas as pd

# Load your YOLOv8 model
model = YOLO('weights/bestV8.pt')

st.title("Chess Detection with Occupancy Grid")

# Sidebar settings
st.sidebar.title("Settings")
conf_threshold = st.sidebar.slider("Confidence threshold", min_value=0.0, max_value=1.0, value=0.50, step=0.05)
grid_rows = grid_cols = 8

# Initialize board and move history
if 'board' not in st.session_state:
    st.session_state.board = chess.Board()
    st.session_state.move_history = []
    st.session_state.chessboard = [
        ['br', 'bn', 'bb', 'bk', 'bq', 'bb', 'bn', 'br'],
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
                print("changed ", chessboard[row][col])
                if new_board_status[row][col] == 'empty' and previous_board_status[row][col] != 'empty':
                    print("start ", chessboard[row][col])
                    move['start'] = (row, col)
                    move['piece'] = chessboard[row][col]
                elif previous_board_status[row][col] == 'empty' and new_board_status[row][col] != 'empty':
                    move['end'] = (row, col)
                    print("end ", chessboard[row][col])
                elif previous_board_status[row][col] == 'black' and new_board_status[row][col] == 'white':
                    print("end + eli ", chessboard[row][col])
                    move['end'] = (row, col)
                    move['eliminated'] = chessboard[row][col]
                elif previous_board_status[row][col] == 'white' and new_board_status[row][col] == 'black':
                    print("end + eli ", chessboard[row][col])
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


stframe = st.empty() 

capture = st.empty()

col1, col2 = st.columns(2)
with col1:
    det_out = st.empty()
with col2:
    occ_out = st.empty()

det_boxes = st.empty()
summary_out = st.empty()
board_status = st.empty()
initial_status = st.empty()

# Process the frame and create occupancy map
def process_frame(frame):
    spaces = create_grid(grid_rows, grid_cols)
    results = model.predict(source=frame, conf=conf_threshold)

    xyxy = results[0].boxes.xyxy
    if len(xyxy) == 0:
        return None
    new_shape, lm, bm = get_frame_new_shape(xyxy, frame.shape)

    # Perform YOLO prediction
    boxes = results[0].boxes.xyxy.cpu().numpy() if results else []
    predicted_classes = results[0].boxes.cls
    class_names = model.names
    predicted_class_names = [class_names[int(cls_idx)] for cls_idx in predicted_classes]    

    occupancy = map_detections_to_spaces(boxes, spaces, predicted_class_names, new_shape, grid_rows, grid_cols, lm, bm)
    new_board_status = map_occupancy_to_board_status(occupancy)

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

    # Visualize detections
    detection_vis = results[0].plot() if results else frame
    occ_map = create_occupancy_map(occupancy, grid_rows, grid_cols)

    # Display frames and results
    det_out.image(detection_vis, channels="BGR", use_container_width=True)
    occ_out.image(occ_map, use_container_width=True)
    det_boxes.write(f"Detected boxes: {len(results[0].boxes.cls)} | Missing cells: {64 - len(results[0].boxes.cls)}")

    summary = {
        "Total Spaces": len(spaces),
        "Initial": sum(1 for s in occupancy.values() if s == "initial"),
        "Empty": sum(1 for s in occupancy.values() if s == "empty"),
        "Black": sum(1 for s in occupancy.values() if s == "black"),
        "White": sum(1 for s in occupancy.values() if s == "white"),
    }
    summary_out.write(summary)
    # Display move table
    st.write("### Move History Table")
    st.dataframe(move_table)
    board_status.text("\n".join(str(row) for row in new_board_status))

# Live webcam feed
def live_camera_feed():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Unable to access the camera.")
        return
    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture frame.")
        # Display the live video frame
        stframe.image(frame, channels="BGR", use_container_width=True)


def fr_capture():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Unable to access the camera.")
        return

    ret, frame = cap.read()
    if not ret:
        st.error("Failed to capture frame.")
    else:
        # Display the live video frame
        process_frame(frame)


if capture.button("Capture"):
    fr_capture()

live_camera_feed()