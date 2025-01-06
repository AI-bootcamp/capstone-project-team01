import streamlit as st
import cv2
import tempfile
from ultralytics import YOLO
from frame_processing_functions import *
import chess
import chess.svg
from io import BytesIO
import pandas as pd
from chess_functions import *

# Load YOLO model
model = YOLO('weights/bestV8.pt')

# Initialize board and move history
if 'board' not in st.session_state:
    start_game()

board = st.session_state.board
move_history = st.session_state.move_history
white_moves = st.session_state.white_moves
black_moves = st.session_state.black_moves

# Confidence Threshold
st.session_state.conf_threshold = 0.7


# Placeholders
st.title("Chess Game with Detection")
warning = st.empty() 
result_announcement = st.empty()
col1, col2 = st.columns(2)
with col1:
    det_out = st.empty()
with col2:
    image_out = st.empty()

white_sec, black_sec, board_sec = st.columns(3)
with white_sec:
    st.write("### White Player Moves")
    white_moves_placeholder = st.dataframe(white_moves)
with black_sec:
    st.write("### Black Player Moves")
    black_moves_placeholder = st.dataframe(black_moves)
with board_sec:
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
        
        new_board_status = order_detections(boxes, predicted_class_names)
        move = detect_move(st.session_state.previous_board_status, new_board_status, st.session_state.chessboard)

        if 'start' in move and 'end' in move:
            start_square = f"{chr(97 + move['start'][1])}{8 - move['start'][0]}"
            end_square = f"{chr(97 + move['end'][1])}{8 - move['end'][0]}"
            piece_name = piece_names.get(move['piece'], 'Unknown')
            eliminated_piece = piece_names.get(move.get('eliminated', ''), '')

            chess_move = chess.Move.from_uci(f"{start_square}{end_square}")
            move_data = [piece_name, start_square, end_square, eliminated_piece]

            if chess_move in board.legal_moves:
                warning.empty()

                if move['piece'].startswith('w'):
                    white_moves.loc[len(white_moves)] = move_data
                else:
                    black_moves.loc[len(black_moves)] = move_data

                st.session_state.chessboard = update_chessboard(move, st.session_state.chessboard)

                st.session_state.previous_board_status = new_board_status

                white_moves_placeholder.dataframe(white_moves)
                black_moves_placeholder.dataframe(black_moves)
            
                board.push(chess_move)
                board_svg_placeholder.markdown(update_board_display(board), unsafe_allow_html=True)

                # Check win and display message
                status, message = check_win_condition(board)
                if status == "success":
                    result_announcement.success(message)
                elif status == "warning":
                    result_announcement.warning(message)
            else:
                reason = explain_illegal_move(board, chess_move)
                warning.warning(f"Move {chess_move} is an illegal move: {reason}")


# Upload and process image
uploaded_file = st.file_uploader("Upload a chess image", type=["jpg", "jpeg", "png", "bmp"])
if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_image:
        temp_image.write(uploaded_file.read())
        temp_image_path = temp_image.name

        image_out.image(temp_image_path, caption="Uploaded Image", use_container_width=True)

        process_image(temp_image_path)

# Button to export to PDF
if st.button("Export Move Tables to PDF"):
    pdf = export_to_pdf(white_moves, black_moves)
    st.download_button(
        label="Download Move History as PDF",
        data=pdf,
        file_name="chess_move_history.pdf",
        mime="application/pdf"
    )
