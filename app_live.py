import streamlit as st
import cv2
from ultralytics import YOLO
from chess_functions import *
from frame_processing_functions import *
import chess
import chess.svg

model = YOLO('weights/bestV9.pt')

st.set_page_config(page_title="Mid Chess Game Detection", page_icon="♟️")

# Initialize variables
if 'board' not in st.session_state:
    start_game()

board = st.session_state.board
move_history = st.session_state.move_history
white_moves = st.session_state.white_moves
black_moves = st.session_state.black_moves

st.session_state.conf_threshold = 0.7

# Streamlit Placeholders
st.title("Chessgame history detection")
warning_placeholder = st.empty() 
result_announcement = st.empty()
frame_col, detection_col2 = st.columns(2)
with frame_col:
    frame_placeholder = st.empty() 
with detection_col2:
    detection_placeholder = st.empty()

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
 
prev_col, new_col = st.columns(2)
with prev_col:
    prev_status_placeholder = st.empty() 
with new_col:
    new_status_placeholder = st.empty()

# Display the Initial board
board_svg_placeholder.markdown(update_board_display(board), unsafe_allow_html=True)

# Process frame function
def process_frame(frame):
    results = model.predict(source=frame, conf=st.session_state.conf_threshold)
    boxes_no = len(results[0].boxes.xyxy)

    # Ensuring that only 64 boxes are detected
    if not results or boxes_no == 0:
        warning_placeholder.warning(f'No results!')
        return 
    elif boxes_no > 64:
            if st.session_state.conf_threshold < 0.9:
                st.session_state.conf_threshold += 0.05
            det_boxes_summary.write(f'number of detected boxes {boxes_no}, while expected is 64. new confidence = {st.session_state.conf_threshold}')
            return
    elif boxes_no < 64:
        if st.session_state.conf_threshold > 0.5:
            st.session_state.conf_threshold -= 0.05
        det_boxes_summary.write(f'number of detected boxes {boxes_no}, while expected is 64. new confidence = {st.session_state.conf_threshold}')
        return
    
    # Display the detection
    det_boxes_summary.write(f"Detected boxes: {boxes_no} | Missing cells: {64 - boxes_no}")
    detection_placeholder.image(results[0].plot(), channels="BGR", use_container_width=True)

    # Get New board status [white, black, empty]
    boxes = results[0].boxes.xyxy.cpu().numpy()
    predicted_classes = results[0].boxes.cls
    class_names = model.names
    predicted_class_names = [class_names[int(cls_idx)] for cls_idx in predicted_classes]    

    new_board_status = order_detections(boxes, predicted_class_names)
    
    # Display Board status if there are issues
    prev_status_placeholder.pyplot(display_board_status(st.session_state.previous_board_status))
    new_status_placeholder.pyplot(display_board_status(new_board_status))

    # Get the move using status changes
    move = detect_move(st.session_state.previous_board_status, new_board_status, board)

    # For Move Suggestion Feature
    is_suggested = move.get('is_suggested', False)
    move_warning = move.get('warning', '')
    if move_warning:
        suggested_move.write(move_warning)
        return
    
    # Validate the move
    if 'start' in move and 'end' in move:
        start_square = move['start']
        end_square = move['end']
        piece_name = move['piece']
        eliminated_piece = move.get('eliminated', '')
        castle = move.get('castle', '')

        chess_move = chess.Move.from_uci(f"{start_square}{end_square}")
        move_data = [piece_name, start_square, end_square, eliminated_piece, castle]

        # For Move Suggestion Feature
        if is_suggested:
            suggested_move.write(f'suggested move is: {move_data}')
            return
        suggested_move.empty()

        # Add move if legal else Display Errors
        if chess_move in board.legal_moves:
            warning_placeholder.empty()

            if board.turn: # True if white
                white_moves.loc[len(white_moves)] = move_data
            else:
                black_moves.loc[len(black_moves)] = move_data


            st.session_state.previous_board_status = new_board_status

            white_moves_placeholder.dataframe(white_moves)
            black_moves_placeholder.dataframe(black_moves)
        
            board.push(chess_move)
            board_svg_placeholder.markdown(update_board_display(board), unsafe_allow_html=True)
            
            if move.get('castle', ''):
                st.session_state.previous_board_status = map_board_to_board_status(board)

            # Check win and display message
            status, message = check_win_condition(board)
            if status == "success":
                result_announcement.success(message)
            elif status == "warning":
                result_announcement.warning(message)
        else:
            reason = explain_illegal_move(board, chess_move)
            warning_placeholder.warning(f"Move {chess_move} is an illegal move: {reason}")

# Live webcam feed
def live_camera_feed():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Unable to access the camera.")
        return

    skip_frame = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                warning_placeholder.warning("Failed to capture frame. Retrying...")
                continue

            # Process every 30th frame
            if skip_frame % 30 == 0:
                # Display the live video frame
                frame_placeholder.image(frame, channels="BGR", use_container_width=True)
                process_frame(frame)

            skip_frame += 1
    except Exception as e:
        st.error(f"An error occurred: {e}")
    finally:
        cap.release() 

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