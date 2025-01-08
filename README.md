# Chess Game Detection and Move Tracking

## Project Overview
This project implements a real-time chess game move detection using YOLO (You Only Look Once), OpenCV and Streamlit. The application utilizes a webcam feed to analyze and recognize chessboard movements, tracks the game progress, and generates a move history and export it to PDF file . Additionally, suggest moves to user using the Stockfish chess engine.

## Features
- **Real-time Chess Detection**: Use the webcam to capture and analyze chessboard movements.
- **YOLO Model Integration**: Detects chess piece positions using a pre-trained YOLOv8 model.
- **Chessboard Rendering**: Displays the current state of the chessboard using SVG rendering.
- **Move History Tracking**: Logs each move made by white and black players.
- **Illegal Move Detection**: Alerts if an illegal move is detected.
- **Move Suggestion**: Provide the user with possible move options for their pieces.
- **Move Evaluation**: Evaluates the quality of a move by comparing the board state before and after the move. 
- **Export to PDF**: Download the move history as a PDF file.

## Technologies and Libraries
<p align="center">
    <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white">
    <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white">
    <img src="https://img.shields.io/badge/YOLO-00FFFF?style=for-the-badge&logo=yolo&logoColor=black">
    <img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white">
    <img src="https://img.shields.io/badge/Python--Chess-FFD43B?style=for-the-badge&logo=python&logoColor=darkgreen">
    <img src="https://img.shields.io/badge/ReportLab-FF8C00?style=for-the-badge">
    <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white">
</p>

## Project Directory Structure
```
📂 Chess-Detection-Tracking
│
├── 📁 .github
│
├── 📁 __pycache__
│
├── 📁 model testing
│
├── 📁 pages
│
├── 📁 stockfish
│
├── 📁 training images
│
├── 📁 weights
│   └── bestV9.pt
│
├── 📁 helpers
│   └── __init__.py
│
├── Uapp.py
├── app.py
├── app_live.py
├── app_live_castling.py
├── chess_functions.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Dataset Information
The dataset used for training this model was collected personally by capturing images of a chessboard under various lighting conditions and angles. Each image was annotated manually to segment the chessboard into three classes:
- **White (Square)**
- **Black (Square)**
- **Empty (Square)**

### Dataset Overview
- **Total Images**: 96
- **Train Set**: 81 images (84%)
- **Validation Set**: 10 images (10%)
- **Test Set**: 5 images (5%)

### Preprocessing
- **Auto-Orientation**: Applied
- **Resize**: Images resized to 640x640

### Augmentations
- **Outputs per Training Example**: 3
- **Flip**: Horizontal, Vertical
- **Grayscale**: Apply to 15% of images
- **Exposure**: Between -10% and +10%

This dataset ensures robust detection of chess pieces and empty squares by training the model across different orientations and brightness levels.

## Installation and Setup
### Prerequisites
Ensure you have Python 3.8 or higher installed.

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Clone the Repository
```bash
git clone <repository-url>
cd <repository-folder>
```

### Model Weights
Place the YOLOv11 model weights (e.g., `bestV9.pt`) in a `weights` directory:
```
weights/bestV9.pt
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
model = YOLO('weights/bestV9.pt')
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


### Move Evaluation

```python
def get_move_evaluation(eval_before, eval_after):
    ep_before = calculate_expected_points(eval_before)
    ep_after = calculate_expected_points(eval_after)
    ep_delta = abs(ep_after - ep_before)

```
The `get_move_evaluation` function evaluates the quality of a move by comparing the board state before and after the move.

- It calculates the expected points (`ep_before` and `ep_after`).
- The difference between the two (`ep_delta`) is used to classify the move.
- The function iterates through predefined classification thresholds to label the move as beneficial, neutral, or detrimental.
- The classification is printed and returned to provide immediate feedback on the move's impact on the player's position.


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


  ## Developers
- Lamees Aloqlan
- Sama Aldawayhie
- Basel Felemban
- Tariq Alshammari
