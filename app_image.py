import streamlit as st
import cv2
import tempfile
from ultralytics import YOLO
from helpers import *

# Load your YOLOv8 model
model = YOLO('weights/bestV8.pt')
board_model = YOLO('weights/best_board.pt')

st.title("Chess Detection with Occupancy Grid")

# Sidebar settings
st.sidebar.title("Settings")
conf_threshold = st.sidebar.slider("Confidence threshold", min_value=0.0, max_value=1.0, value=0.05, step=0.05)
grid_rows = grid_cols = 8

from PIL import Image

def crop_image(image, xyxy):
    """
    Crops the input image based on the bounding box of the detected object.

    Parameters:
        image (PIL.Image.Image): The input image.
        results (object): The detection results containing bounding box coordinates.

    Returns:
        PIL.Image.Image: The cropped image.
    """
    # Assuming `results` contains bounding box coordinates in the form:
    # [xmin, ymin, xmax, ymax]
    min_x = xyxy[:, 0].min().item() - 10
    max_x = xyxy[:, 2].max().item() + 10
    min_y = xyxy[:, 1].min().item() - 10
    max_y = xyxy[:, 3].max().item() + 10

    # Crop the image using the bounding box
    cropped_image = image.crop((min_x, min_y, max_x, max_y))

    return cropped_image


# Process the image
def process_image(imagePath):
    spaces = create_grid(grid_rows, grid_cols)
    results = board_model.predict(source=imagePath, conf=conf_threshold)
    cropping = st.empty()
    detection_vis = results[0].plot()
    cropping.image(detection_vis, channels="BGR", use_container_width=True)

    img = Image.open(imagePath)
    croppedImage = crop_image(img, results[0].boxes.xyxy)

    results = model.predict(source=croppedImage, conf=conf_threshold)

    new_shape, lm, bm = get_new_shape(results[0].boxes.xyxy, croppedImage.size)

    det_out = st.empty()
    occ_out = st.empty()
    det_boxes = st.empty()
    summary_out = st.empty()
    board_status = st.empty()

    # Perform YOLO prediction
    boxes = results[0].boxes.xyxy.cpu().numpy() if results else []

    predicted_classes = results[0].boxes.cls
    class_names = model.names
    predicted_class_names = [class_names[int(cls_idx)] for cls_idx in predicted_classes]    

    occupancy = map_detections_to_spaces(boxes, spaces, predicted_class_names, new_shape, grid_rows, grid_cols, lm, bm)

    new_board_status = map_occupancy_to_board_status(occupancy)


    # Visualize detections
    detection_vis = results[0].plot() if results else img

    # Create the occupancy map
    occ_map = create_occupancy_map(occupancy, grid_rows, grid_cols)

    # Display the frames
    det_out.image(detection_vis, channels="BGR", use_container_width=True)
    occ_out.image(occ_map, use_container_width=True)

    det_boxes.write(f"Number of detected boxes = {len(results[0].boxes.cls)} Missing cells = {64 - len(results[0].boxes.cls)}")
    # Display summary
    summary = {
        "Total Spaces": len(spaces),
        "Initial": sum(1 for s in occupancy.values() if s == "initial"),
        "Empty": sum(1 for s in occupancy.values() if s == "empty"),
        "Black": sum(1 for s in occupancy.values() if s == "black"),
        "White": sum(1 for s in occupancy.values() if s == "white"),
    }
    summary_out.write(summary)
    board_status.text("\n".join(str(row) for row in new_board_status))



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

