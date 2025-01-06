import streamlit as st
import cv2
from ultralytics import YOLO
from chessFunction_castling import *
from helpers import *
import chess
import chess.svg

# Load your YOLOv8 model
model = YOLO('weights/bestV9.pt')

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
suggested_move = st.empty()

white_sec, black_sec, board_sec = st.columns(3)
with white_sec:
    st.write("### White Player Moves")
    white_moves_placeholder = st.dataframe(white_moves)
with black_sec:
    st.write("### Black Player Moves")
    black_moves_placeholder = st.dataframe(black_moves)
with board_sec:
    board_svg_placeholder = st.empty()
 
col21, col22 = st.columns(2)
with col1:
    prev_status_placeholder = st.empty() 
with col2:
    new_status_placeholder = st.empty()

board_svg_placeholder.markdown(update_board_display(board), unsafe_allow_html=True)

def process_frame(frame):
    results = model.predict(source=frame, conf=st.session_state.conf_threshold)
    xyxy = results[0].boxes.xyxy

    if not results or len(xyxy) == 0:
        warning.warning(f'No results!')
        return 
    elif len(xyxy) > 64:
        if st.session_state.conf_threshold < 0.9:
            st.session_state.conf_threshold += 0.05
        det_boxes_summary.write(f'number of detected boxes {len(xyxy)}, while expected is 64. new confidence = {st.session_state.conf_threshold}')
        return
    elif len(xyxy) < 64:
        if st.session_state.conf_threshold > 0.5:
            st.session_state.conf_threshold -= 0.05
        det_boxes_summary.write(f'number of detected boxes {len(xyxy)}, while expected is 64. new confidence = {st.session_state.conf_threshold}')
        return

    boxes = results[0].boxes.xyxy.cpu().numpy()
    predicted_classes = results[0].boxes.cls
    class_names = model.names
    predicted_class_names = [class_names[int(cls_idx)] for cls_idx in predicted_classes]    

    det_boxes_summary.write(f"Detected boxes: {len(xyxy)} | Missing cells: {64 - len(xyxy)}")

    stdetected_boxes.image(results[0].plot(), channels="BGR", use_container_width=True)

    new_board_status = order_detections(boxes, predicted_class_names)
    
    # Display Board status if there are issues
    prev_status_placeholder.text("\n".join(str(row) for row in st.session_state.previous_board_status))
    new_status_placeholder.text("\n".join(str(row) for row in new_board_status))

    # Detect moves (including castling)
    move = detect_move(st.session_state.previous_board_status, new_board_status, st.session_state.chessboard, board)
    is_suggested = move.get('is_suggested', False)

    if 'start' in move and 'end' in move:
        start_square = f"{chr(97 + move['start'][1])}{8 - move['start'][0]}"
        end_square = f"{chr(97 + move['end'][1])}{8 - move['end'][0]}"
        piece_name = piece_names.get(move['piece'], 'Unknown')
        eliminated_piece = piece_names.get(move.get('eliminated', ''))
        
        # Get Castling Information
        castling_type = move.get('castling', '')  # O-O or O-O-O
        rook_from = move.get('rook_start', '')
        rook_to = move.get('rook_end', '')

        # Update move data with castling details
        move_data = [piece_name, start_square, end_square, eliminated_piece, castling_type, rook_from, rook_to]

        if is_suggested:
            suggested_move.write(f'suggested move is: {move_data}')
            return
        suggested_move.empty()

        # If the move is legal, push it to the board
        chess_move = chess.Move.from_uci(f"{start_square}{end_square}")
        if chess_move in board.legal_moves:
            warning.empty()

            # Record the move with castling info
            if move['piece'].startswith('w'):
                white_moves.loc[len(white_moves)] = move_data
            else:
                black_moves.loc[len(black_moves)] = move_data

            # Update the board and session state
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

            # Display the live video frame
            stframe.image(frame, channels="BGR", use_container_width=True)

            # Process every 10th frame
            if skip_frame % 10 == 0:
                process_frame(frame)

            skip_frame += 1
    except Exception as e:
        st.error(f"An error occurred: {e}")
    finally:
        cap.release()  # Ensure the camera is released when done

# Button to export to PDF
if st.button("Export Move Tables to PDF"):
    pdf = export_to_pdf(white_moves, black_moves)
    st.download_button(
        label="Download Move History as PDF",
        data=pdf,
        file_name="chess_move_history.pdf",
        mime="application/pdf"
    )

live_camera_feed()