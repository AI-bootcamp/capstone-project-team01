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
import logging

# Load YOLO model
try:
    model = YOLO('weights/bestV5.pt')
except Exception as e:
    st.error(f"Failed to load YOLO model: {str(e)}")
    st.stop()

# Initialize board and move history
if 'board' not in st.session_state:
    st.session_state.board = chess.Board()
    st.session_state.move_history = []

board = st.session_state.board
move_history = st.session_state.move_history

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
    board_svg_placeholder.markdown(
        f'<img src="data:image/svg+xml;base64,{encoded_svg}" width="400"/>',
        unsafe_allow_html=True
    )

update_board_display()

# Process the image to detect chess pieces
def process_image(imagePath):
    spaces = create_grid(grid_rows, grid_cols)
    results = model.predict(source=imagePath, conf=conf_threshold)
    img = cv2.imread(imagePath)

    if results and results[0].boxes.xyxy.size != 0:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        predicted_classes = results[0].boxes.cls
        class_names = model.names
        predicted_class_names = [class_names[int(cls)] for cls in predicted_classes]

        # Map detections to board
        new_shape, lm, bm = get_new_shape(results[0].boxes.xyxy, img.shape[:2])
        occupancy = map_detections_to_spaces(
            boxes, spaces, predicted_class_names, new_shape,
            grid_rows, grid_cols, lm, bm
        )
        new_board_status = map_occupancy_to_board_status(occupancy)

        # Log board status to console (for debugging)
        logging.info(f"Detected Board Status: {new_board_status}")

        # Update board based on detection
        square_index = 0
        for row in new_board_status:
            if isinstance(row, list):
                for piece in row:
                    square = chess.SQUARES[square_index]
                    square_index += 1

                    if piece == 'white':
                        board.set_piece_at(square, chess.Piece(chess.PAWN, chess.WHITE))
                    elif piece == 'black':
                        board.set_piece_at(square, chess.Piece(chess.PAWN, chess.BLACK))
                    elif piece == 'empty':
                        board.remove_piece_at(square)
            else:
                st.error(f"Unexpected row format: {row}")

        update_board_display()
        st.success("Chessboard processed successfully. The board has been updated!")
    else:
        st.warning("No chess pieces detected. Please upload a clearer image.")

# Upload and process image
uploaded_file = st.file_uploader(
    "Upload a chess image", type=["jpg", "jpeg", "png", "bmp"]
)
if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_image:
        temp_image.write(uploaded_file.read())
        temp_image_path = temp_image.name

        # Hide uploaded image preview
        st.success("Image uploaded successfully!")

        if st.button("Process Image"):
            process_image(temp_image_path)

# Display move history
st.write("### Move History")
if move_history:
    for i, mv in enumerate(move_history, start=1):
        st.write(f"{i}. {mv}")
else:
    st.write("No moves recorded yet. Upload an image to detect the board or process pieces.")

# Export move history to PDF
if move_history and st.button("Export Move History to PDF"):
    pdf = BytesIO()
    c = canvas.Canvas(pdf)
    
    c.setFont("Helvetica", 14)
    c.drawString(100, 800, "Chess Game - Move History")
    
    y = 780
    c.setFont("Helvetica", 12)
    for i, move in enumerate(move_history, start=1):
        y -= 20
        c.drawString(100, y, f"{i}. {move}")
        
    c.save()
    pdf.seek(0)

    st.download_button(
        label="Download Move History as PDF",
        data=pdf,
        file_name="chess_move_history.pdf",
        mime="application/pdf"
    )
else:
    st.write("No moves to export yet.")
