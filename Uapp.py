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
model = YOLO('weights/bestV8.pt')

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
    st.session_state.white_moves = pd.DataFrame(columns=["Piece", "From", "To", "Eliminated"])
    st.session_state.black_moves = pd.DataFrame(columns=["Piece", "From", "To", "Eliminated"])

piece_names = {
    'wr': 'Rook', 'wn': 'Knight', 'wb': 'Bishop', 'wq': 'Queen', 'wk': 'King', 'wp': 'Pawn',
    'br': 'Rook', 'bn': 'Knight', 'bb': 'Bishop', 'bq': 'Queen', 'bk': 'King', 'bp': 'Pawn'
}

board = st.session_state.board
move_history = st.session_state.move_history
white_moves = st.session_state.white_moves
black_moves = st.session_state.black_moves

st.title("Chess Game with Detection")

# Sidebar settings
st.sidebar.title("Settings")
conf_threshold = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.6, 0.05)
grid_rows = grid_cols = 8

col1, col2 = st.columns(2)
with col1:
    det_out = st.empty()
with col2:
    image_out = st.empty()
    


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
                elif previous_board_status[row][col] != new_board_status[row][col]:
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
    # spaces = create_grid(grid_rows, grid_cols)
    results = model.predict(source=imagePath, conf=conf_threshold)
    # img = cv2.imread(imagePath)

    if results:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        predicted_classes = results[0].boxes.cls
        class_names = model.names
        predicted_class_names = [class_names[int(cls)] for cls in predicted_classes]

        # new_shape, lm, bm = get_new_shape(boxes, img.shape[:2])
        # occupancy = map_detections_to_spaces(boxes, spaces, predicted_class_names, new_shape, grid_rows, grid_cols, lm, bm)
        # new_board_status = map_occupancy_to_board_status(occupancy)
        
        detection_vis = results[0].plot()
        det_out.image(detection_vis, channels="BGR", use_container_width=True)
        
        new_board_status = order_detections(boxes, predicted_class_names)

        move = detect_move(st.session_state.previous_board_status, new_board_status, st.session_state.chessboard)
        st.session_state.previous_board_status = new_board_status

        if 'start' in move and 'end' in move:
            start_square = f"{chr(97 + move['start'][1])}{8 - move['start'][0]}"
            end_square = f"{chr(97 + move['end'][1])}{8 - move['end'][0]}"
            piece_name = piece_names.get(move['piece'], 'Unknown')
            eliminated_piece = piece_names.get(move.get('eliminated', ''), '')

            chess_move = chess.Move.from_uci(f"{start_square}{end_square}")
            move_data = [piece_name, start_square, end_square, eliminated_piece]
            if move['piece'].startswith('w'):
                white_moves.loc[len(white_moves)] = move_data
            else:
                black_moves.loc[len(black_moves)] = move_data

            st.session_state.chessboard = update_chessboard(move, st.session_state.chessboard)

            if chess_move in board.legal_moves:
                board.push(chess_move)
                update_board_display()


# Export move tables to PDF
def export_to_pdf():
    pdf = BytesIO()
    c = canvas.Canvas(pdf)
    c.setFont("Helvetica", 16)
    c.drawString(200, 800, "Chess Game Move History")
    c.setFont("Helvetica", 14)

    y = 760
    c.drawString(100, y, "White Player Moves:")
    y -= 20
    for index, row in white_moves.iterrows():
        c.drawString(100, y, f"{row['Piece']} from {row['From']} to {row['To']} (Eliminated: {row['Eliminated']})")
        y -= 20

    y -= 40
    c.drawString(100, y, "Black Player Moves:")
    y -= 20
    for index, row in black_moves.iterrows():
        c.drawString(100, y, f"{row['Piece']} from {row['From']} to {row['To']} (Eliminated: {row['Eliminated']})")
        y -= 20

    c.save()
    pdf.seek(0)
    st.download_button(
        label="Download Move History as PDF",
        data=pdf,
        file_name="chess_move_history.pdf",
        mime="application/pdf"
    )

# Upload and process image
uploaded_file = st.file_uploader("Upload a chess image", type=["jpg", "jpeg", "png", "bmp"])
if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_image:
        temp_image.write(uploaded_file.read())
        temp_image_path = temp_image.name

        image_out.image(temp_image_path, caption="Uploaded Image", width=700)

        if st.button("Process Image"):
            process_image(temp_image_path)

# Display move tables
st.write("### White Player Moves")
st.dataframe(white_moves)
st.write("### Black Player Moves")
st.dataframe(black_moves)

# Button to export to PDF
if st.button("Export Move Tables to PDF"):
    export_to_pdf()
