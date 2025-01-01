import streamlit as st
import cv2
import numpy as np
import tempfile
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

# Create a grid for spaces
def create_grid(grid_rows, grid_cols):
    spaces = []
    for i in range(grid_rows):
        for j in range(grid_cols):
            spaces.append(f"{chr(65 + i)}{j + 1}")
    return spaces

# Function to get the board coordinates
def get_board_coordinates(xyxy):
    min_x = xyxy[:, 0].min().item()  # Use .min() to find the minimum and .item() to get scalar
    max_x = xyxy[:, 2].max().item()  # Use .max() for the maximum value
    min_y = xyxy[:, 1].min().item()
    max_y = xyxy[:, 3].max().item()
    return min_x - 5, min_y - 5, max_x + 5, max_y + 5

# Map detections to cells
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
        
        # Map to a space identifier if within bounds
        space = f"{chr(65 + row)}{col + 1}"  # Convert row index to letter and col index to number
        occupancy[space] = classes[index]
        
        index += 1
    
    return occupancy


# Create the occupancy grid visualization
def create_occupancy_map(occupancy, grid_rows, grid_cols, map_shape = (480, 640, 3)):
    occ_map = np.full(map_shape, (25, 25, 75), dtype=np.uint8)

    for i, row in enumerate(range(65, 65 + grid_rows)):
        for j in range(grid_cols):
            space = f"{chr(row)}{j + 1}"
            x1 = 10 + j * (map_shape[1] - 20) // grid_cols
            y1 = 10 + i * (map_shape[0] - 20) // grid_rows
            x2 = 10 + (j + 1) * (map_shape[1] - 20) // grid_cols
            y2 = 10 + (i + 1) * (map_shape[0] - 20) // grid_rows

            # Define a mapping of class to color
            class_to_color = {
                "empty": (0, 255, 0),    # Green
                "black": (0, 0, 0),      # Black
                "white": (255, 0, 255),# White
                "initial": (255, 100, 0) # Yellow
            }
            color = class_to_color.get(occupancy[space], (128, 128, 128))  # Default color

            cv2.rectangle(occ_map, (x1, y1), (x2, y2), color, -1)
            cv2.putText(occ_map, space, (x1 + 5, y1 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return occ_map

# Get the board
def get_board(imagePath):
    image = cv2.imread(imagePath)
    if image is None:
        raise ValueError(f"Unable to load image from path: {imagePath}")

    results = model.predict(source=image, conf=conf_threshold)

    min_x, min_y, max_x, max_y = get_board_coordinates(results[0].boxes.xyxy)
    img = Image.open(imagePath)

    # Crop the image
    return img.crop((min_x, min_y, max_x, max_y))

# Process the image
def process_image(imagePath):
    spaces = create_grid(grid_rows, grid_cols)
    board = get_board(imagePath)

    # Output placeholders
    det_out = st.empty()
    occ_out = st.empty()
    summary_out = st.empty()

    # Perform YOLO prediction
    results = model.predict(source=board, conf=conf_threshold)
    boxes = results[0].boxes.xyxy.cpu().numpy() if results else []

    predicted_classes = results[0].boxes.cls
    class_names = model.names
    predicted_class_names = [class_names[int(cls_idx)] for cls_idx in predicted_classes]    

    occupancy = map_detections_to_spaces(boxes, spaces, predicted_class_names, board.size, grid_rows, grid_cols)

    # Visualize detections
    detection_vis = results[0].plot() if results else board

    # Create the occupancy map
    occ_map = create_occupancy_map(occupancy, grid_rows, grid_cols)

    # Display the frames
    det_out.image(detection_vis, channels="BGR", use_container_width=True)
    occ_out.image(occ_map, use_container_width=True)

    # Display summary
    summary = {
        "Total Spaces": len(spaces),
        "Initial": sum(1 for s in occupancy.values() if s == "initial"),
        "Empty": sum(1 for s in occupancy.values() if s == "empty"),
        "Black": sum(1 for s in occupancy.values() if s == "black"),
        "White": sum(1 for s in occupancy.values() if s == "white"),
    }
    summary_out.write(summary)


# Upload and process image
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png", "bmp"])
if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_image:
        temp_image.write(uploaded_file.read())
        temp_image_path = temp_image.name

        # Display the uploaded image
        st.image(temp_image_path, caption="Uploaded Image",  use_container_width=True)

        if st.button("Process Image"):
            process_image(temp_image_path)

