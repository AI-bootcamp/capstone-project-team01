import numpy as np
import cv2

# Create a grid for spaces (reversed order)
def create_grid(grid_rows, grid_cols):
    spaces = []
    for i in range(grid_rows):
        for j in range(grid_cols):
            spaces.append(f"{chr(72 - i)}{j + 1}")  # Start from 'H' (72 in ASCII)
    return spaces

# Function to get the board coordinates
def get_board_coordinates(xyxy):
    min_x = xyxy[:, 0].min().item()
    max_x = xyxy[:, 2].max().item()
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
def get_board_from_frame(frame, results):
    if results:
        detection_original = results[0].plot()
        boxes = results[0].boxes.xyxy.cpu().numpy()
        min_x, min_y, max_x, max_y = get_board_coordinates(boxes)
        cropped_frame = None
        # Crop the image to focus on the board
        cropped_frame = frame[int(min_y):int(max_y), int(min_x):int(max_x)]
        return cropped_frame, detection_original
    else:
        return None