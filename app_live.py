import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

# Load your YOLOv8 model
model = YOLO('weights/bestV2.pt')

st.title("Chess Detection with Occupancy Grid")

# Sidebar settings
st.sidebar.title("Settings")
conf_threshold = st.sidebar.slider("Confidence threshold", min_value=0.0, max_value=1.0, value=0.05, step=0.05)
grid_rows = 8
grid_cols = 8

# Create a grid for spaces (reversed order)
def create_grid(grid_rows, grid_cols):
    spaces = []
    for i in range(grid_rows):
        for j in range(grid_cols):
            spaces.append(f"{chr(72 - i)}{j + 1}")  # Start from 'H' (72 in ASCII)
    return spaces

# Function to get the board coordinates
def get_board_coordinates(xyxy):
    min_x = xyxy[:, 0].min().item()  # Use .min() to find the minimum and .item() to get scalar
    max_x = xyxy[:, 2].max().item()  # Use .max() for the maximum value
    min_y = xyxy[:, 1].min().item()
    max_y = xyxy[:, 3].max().item()
    return min_x - 5, min_y - 5, max_x + 5, max_y + 5

# Map detections to cells (reversed grid)
def map_detections_to_spaces(boxes, spaces, classes, frame_shape, grid_rows, grid_cols):
    # Initialize all spaces to "initial"
    occupancy = {space: "initial" for space in spaces}
    index = 0
    
    for box in boxes:
        # Calculate the center of the box
        x_center = (box[0] + box[2]) / 2
        y_center = (box[1] + box[3]) / 2
        
        # Calculate row and column indices
        row = int(y_center * grid_rows / frame_shape[0])
        col = int(x_center * grid_cols / frame_shape[1])
        
        # Clamp row and column indices to valid grid bounds
        row = max(0, min(row, grid_rows - 1))  # Ensure 0 <= row < grid_rows
        col = max(0, min(col, grid_cols - 1))  # Ensure 0 <= col < grid_cols
        
        # Map to a space identifier, reversed for rows
        space = f"{chr(72 - row)}{col + 1}"  # Reverse row index to start from 'H' (72)
        occupancy[space] = classes[index]
        
        index += 1
    
    return occupancy

# Create the occupancy grid visualization
def create_occupancy_map(occupancy, grid_rows, grid_cols, map_shape=(480, 640, 3)):
    occ_map = np.full(map_shape, (25, 25, 75), dtype=np.uint8)

    for i, row in enumerate(range(65, 65 + grid_rows)):
        for j in range(grid_cols):
            space = f"{chr(72 - i)}{j + 1}"  # Reverse row to start from 'H' (72)
            x1 = 10 + j * (map_shape[1] - 20) // grid_cols
            y1 = 10 + i * (map_shape[0] - 20) // grid_rows
            x2 = 10 + (j + 1) * (map_shape[1] - 20) // grid_cols
            y2 = 10 + (i + 1) * (map_shape[0] - 20) // grid_rows

            # Define a mapping of class to color
            class_to_color = {
                "empty": (0, 255, 0),    # Green
                "black": (0, 0, 0),      # Black
                "white": (255, 0, 255),  # White
                "initial": (255, 100, 0) # Yellow
            }
            color = class_to_color.get(occupancy[space], (128, 128, 128))  # Default color

            cv2.rectangle(occ_map, (x1, y1), (x2, y2), color, -1)
            cv2.putText(occ_map, space, (x1 + 5, y1 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return occ_map

# Get the board from the frame
def get_board_from_frame(frame):
    results = model.predict(source=frame, conf=conf_threshold)

    if results:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        min_x, min_y, max_x, max_y = get_board_coordinates(boxes)

        # Crop the image to focus on the board
        cropped_frame = frame[int(min_y):int(max_y), int(min_x):int(max_x)]
        return cropped_frame
    else:
        return None

# Process the frame and create occupancy map
def process_frame(frame, frame_counter):
    spaces = create_grid(grid_rows, grid_cols)

    # Perform YOLO prediction
    results = model.predict(source=frame, conf=conf_threshold)
    boxes = results[0].boxes.xyxy.cpu().numpy() if results else []

    predicted_classes = results[0].boxes.cls
    class_names = model.names
    predicted_class_names = [class_names[int(cls_idx)] for cls_idx in predicted_classes]    

    occupancy = map_detections_to_spaces(boxes, spaces, predicted_class_names, frame.shape, grid_rows, grid_cols)

    # Create the occupancy map
    occ_map = create_occupancy_map(occupancy, grid_rows, grid_cols)

    return occ_map

# Live webcam feed
def live_camera_feed():
    cap = cv2.VideoCapture(1)  
    if not cap.isOpened():
        st.error("Unable to access the camera.")
        return

    stframe = st.empty() 
    stop_button = st.button("Stop Camera Feed")  

    frame_counter = 0  
    occ_map_placeholder = st.empty()  # Placeholder for occ_map

    while True:
        ret, frame = cap.read()

        if not ret:
            st.error("Failed to capture frame.")
            break

        frame_counter += 1

        # Get the chessboard from the frame and process it every 60th frame
        if frame_counter % 60 == 0:
            cropped_board = get_board_from_frame(frame)
            
            if cropped_board is not None:
                occ_map = process_frame(cropped_board, frame_counter)
                occ_map_placeholder.image(occ_map, use_container_width=True)

        # Display the live video frame (to be deleted)
        stframe.image(frame, channels="BGR", use_container_width=True)

        if stop_button:
            cap.release()
            cv2.destroyAllWindows()
            st.write("Camera feed stopped.")
            break

    cap.release()
    cv2.destroyAllWindows()

# Button to start live camera feed
if st.button("Start Live Camera Feed"):
    live_camera_feed()
