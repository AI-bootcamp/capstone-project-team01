# Chess Game Detection and Move Tracking

## Project Overview
This project implements a real-time chess game move detection and tracking system using Streamlit, YOLO (You Only Look Once) object detection, and OpenCV. The application utilizes a webcam feed to analyze and recognize chessboard movements, tracks the game progress, and generates a move history. Additionally, users can export the move history to a PDF file.

## Features
- **Real-time Chess Detection**: Use the webcam to capture and analyze chessboard movements.
- **YOLO Model Integration**: Detects chess piece positions using a pre-trained YOLOv8 model.
- **Chessboard Rendering**: Displays the current state of the chessboard using SVG rendering.
- **Move History Tracking**: Logs each move made by white and black players.
- **Illegal Move Detection**: Alerts if an illegal move is detected.
- **Export to PDF**: Download the move history as a PDF file.

## Technologies and Libraries
- **Python**  
- **Streamlit** – For the user interface and real-time display  
- **YOLO (Ultralytics)** – Object detection model for recognizing chess pieces  
- **OpenCV** – Capturing video feed from webcam  
- **Python-Chess** – Managing chess logic and move legality  
- **ReportLab** – Generating PDFs  
- **Pandas** – Handling and displaying move history  

## Installation and Setup
### Prerequisites
Ensure you have Python 3.8 or higher installed.

### Install Dependencies
```bash
pip install streamlit ultralytics opencv-python-headless chess pandas reportlab
```

### Clone the Repository
```bash
git clone <repository-url>
cd <repository-folder>
```

### Model Weights
Place the YOLOv8 model weights (e.g., `bestV7.pt`) in a `weights` directory:
```
weights/
    bestV7.pt
```

### Run the Application
```bash
streamlit run app.py
```

## How to Use
1. Launch the application by running the Streamlit command above.
2. The webcam feed will display on the main screen.
3. Place a chessboard in front of the camera and make moves.
4. Detected moves are logged in separate tables for white and black players.
5. Click the **Export Move Tables to PDF** button to download the move history.

## Key Components
### Model Loading
```python
model = YOLO('weights/bestV7.pt')
```
The YOLO model is loaded to detect chess pieces from the video frames.

### Chessboard Initialization
```python
if 'board' not in st.session_state:
    st.session_state.board = chess.Board()
```
A chessboard is initialized and stored in session state.

### Frame Processing
```python
def process_frame(frame):
    results = model.predict(source=frame, conf=st.session_state.conf_threshold)
    # Process detection results
```
Frames are processed using YOLO to detect chess pieces' positions.

### Detect and Update Moves
```python
move = detect_move(st.session_state.previous_board_status, new_board_status, st.session_state.chessboard)
```
Detects moves based on the difference between previous and current board states.

### PDF Export
```python
if st.button("Export Move Tables to PDF"):
    export_to_pdf()
```
Exports the move history to a downloadable PDF file.

## Customization
- **Confidence Threshold**: Adjust the confidence threshold dynamically to refine detection.
- **Move Validation**: The system checks for illegal moves and provides warnings.

## Troubleshooting
- Ensure the camera is correctly connected.
- Adjust the chessboard position for better detection.
- Verify that the YOLO model weights are properly configured.

## Future Enhancements
- **Multiple Camera Support**
- **Piece Recognition Improvements**
- **Enhanced Visualization and Analysis**
