import streamlit as st
import cv2
from ultralytics import YOLO
from chess_functions import *
from helpers import *
import chess
import chess.svg
from io import BytesIO
import base64
from reportlab.pdfgen import canvas
import pandas as pd

# Load your YOLOv8 model
model = YOLO('weights/bestV7.pt')

# Initialize variables
if 'board' not in st.session_state:
    start_game()

board = st.session_state.board
move_history = st.session_state.move_history
white_moves = st.session_state.white_moves
black_moves = st.session_state.black_moves

st.session_state.conf_threshold = 0.7

# Placeholders
st.title("Chessgame history detection")
warning = st.empty()
result_announcement = st.empty()
col1, col2 = st.columns(2)
with col1:
    stframe = st.empty()
with col2:
    stdetected_boxes = st.empty()

det_boxes_summary = st.empty()

white_sec, black_sec, board_sec = st.columns(3)
with white_sec:
    st.write("### White Player Moves")
    white_moves_placeholder = st.dataframe(white_moves)
with black_sec:
    st.write("### Black Player Moves")
    black_moves_placeholder = st.dataframe(black_moves)
with board_sec:
    board_svg_placeholder = st.empty()

# Display initial chess board at top right
board_svg = chess.svg.board(board=board)
board_svg_placeholder.markdown(
    f"<div style='display: flex; justify-content: center;'>"
    f"<img src='data:image/svg+xml;base64,{base64.b64encode(board_svg.encode('utf-8')).decode('utf-8')}' "
    f"style='width: 400px;'/></div>",
    unsafe_allow_html=True
)

move_status_placeholder = st.empty()  # Move status placeholder

# Process the frame
def process_frame(frame):
    start_square = ''
    end_square = ''
    piece_name = 'King'
    eliminated_piece = '-'

    results = model.predict(source=frame, conf=st.session_state.conf_threshold)
    xyxy = results[0].boxes.xyxy

    if not results or len(xyxy) == 0:
        move_status_placeholder.write('No results detected!')
        return

    boxes = results[0].boxes.xyxy.cpu().numpy()
    predicted_classes = results[0].boxes.cls
    class_names = model.names
    predicted_class_names = [class_names[int(cls_idx)] for cls_idx in predicted_classes]

    # Ensure correct board size
    new_board_status = order_detections(boxes, predicted_class_names)
    if len(new_board_status) != 8 or any(len(row) != 8 for row in new_board_status):
        warning.warning("Incomplete board detected. Skipping frame.")
        return

    move = detect_move(st.session_state.previous_board_status, new_board_status, st.session_state.chessboard)

    if not move or 'start' not in move or 'end' not in move:
        move_status_placeholder.write("No move detected!")
        return

    # Determine piece color based on start position
    piece_color = 'w' if move['piece'].startswith('w') else 'b'
    current_turn = 'w' if board.turn else 'b'

    if piece_color != current_turn:
        warning.warning(f"Illegal move: It's {current_turn}'s turn, but {piece_color} piece moved.")
        return

    # Castling detection logic
    castling_moves = {
        (7, 4, 7, 6): ('wr', (7, 7), (7, 5)),  # White kingside
        (7, 4, 7, 2): ('wr', (7, 0), (7, 3)),  # White queenside
        (0, 4, 0, 6): ('br', (0, 7), (0, 5)),  # Black kingside
        (0, 4, 0, 2): ('br', (0, 0), (0, 3))   # Black queenside
    }

    if (move['start'][0], move['start'][1], move['end'][0], move['end'][1]) in castling_moves:
        rook_piece, rook_from, rook_to = castling_moves[(move['start'][0], move['start'][1], move['end'][0], move['end'][1])]
        
        # Check if rook is in initial place and alive
        if st.session_state.chessboard[rook_from[0]][rook_from[1]] == rook_piece:
            move['castling'] = (rook_piece, rook_from, rook_to)
            
            # Perform king move for castling
            castling_king_move = chess.Move.from_uci(f"{start_square}{end_square}")
            
            if castling_king_move in board.legal_moves:
                board.push(castling_king_move)  # Push king's castling move
    else:
        # Normal move processing
        st.session_state.chessboard = update_chessboard(move, st.session_state.chessboard)
    
    # Apply move directly to the board
    start_square = f"{chr(97 + move['start'][1])}{8 - move['start'][0]}"
    end_square = f"{chr(97 + move['end'][1])}{8 - move['end'][0]}"
    
    chess_move = chess.Move.from_uci(f"{start_square}{end_square}")
    
    if chess_move in board.legal_moves:
        board.push(chess_move)
        piece_name = piece_names.get(move.get('piece', ''), 'Unknown')
        eliminated_piece = piece_names.get(move.get('eliminated', ''), '-')

        if 'castling' in move:
            rook_piece, rook_start, rook_end = move['castling']
            rook_start_square = f"{chr(97 + rook_start[1])}{8 - rook_start[0]}"
            rook_end_square = f"{chr(97 + rook_end[1])}{8 - rook_end[0]}"
            castling_data = [piece_name, start_square, end_square, '-', 'Rook', rook_start_square, rook_end_square]
            if move['piece'].startswith('w'):
                white_moves.loc[len(white_moves)] = castling_data
            else:
                black_moves.loc[len(black_moves)] = castling_data
        else:
            move_data = [piece_name, start_square, end_square, eliminated_piece, '-', '-', '-']
            if move['piece'].startswith('w'):
                white_moves.loc[len(white_moves)] = move_data
            else:
                black_moves.loc[len(black_moves)] = move_data
    else:
        warning.warning(f"Illegal move detected: {start_square} to {end_square}")
        return

    st.session_state.previous_board_status = new_board_status
    white_moves_placeholder.dataframe(white_moves)
    black_moves_placeholder.dataframe(black_moves)
    board_svg_placeholder.markdown(update_board_display(board), unsafe_allow_html=True)

# Live webcam feed
def live_camera_feed():
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        st.error("Unable to access the camera.")
        return

    skip_frame = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                warning.warning("Failed to capture frame. Retrying...")
                continue

            stframe.image(frame, channels="BGR", use_container_width=True)
            if skip_frame % 30 == 0:
                process_frame(frame)
            skip_frame += 1
    except Exception as e:
        st.error(f"An error occurred: {e}")
    finally:
        cap.release()

live_camera_feed()
