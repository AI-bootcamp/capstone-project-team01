import streamlit as st
import cv2
from ultralytics import YOLO
from helpers import *

# Load your YOLOv8 model
model = YOLO('weights/bestV8.pt')

st.title("Chess Detection with Occupancy Grid")

# Sidebar settings
st.sidebar.title("Settings")
conf_threshold = st.sidebar.slider("Confidence threshold", min_value=0.0, max_value=1.0, value=0.50, step=0.05)
grid_rows = grid_cols = 8


stframe = st.empty() 

capture = st.empty()

col1, col2 = st.columns(2)
with col1:
    det_out = st.empty()
with col2:
    occ_out = st.empty()

det_boxes = st.empty()
summary_out = st.empty()
board_status = st.empty()
initial_status = st.empty()

# Process the frame and create occupancy map
def process_frame(frame):
    spaces = create_grid(grid_rows, grid_cols)
    results = model.predict(source=frame, conf=conf_threshold)

    xyxy = results[0].boxes.xyxy
    if len(xyxy) == 0:
        return None
    new_shape, lm, bm = get_frame_new_shape(xyxy, frame.shape)

    # Perform YOLO prediction
    boxes = results[0].boxes.xyxy.cpu().numpy() if results else []
    predicted_classes = results[0].boxes.cls
    class_names = model.names
    predicted_class_names = [class_names[int(cls_idx)] for cls_idx in predicted_classes]    

    occupancy, grids = map_detections_to_spaces(boxes, spaces, predicted_class_names, new_shape, grid_rows, grid_cols, lm, bm)
    st.write(grids)
    new_board_status = map_occupancy_to_board_status(occupancy)

    # Visualize detections
    detection_vis = results[0].plot() if results else frame
    occ_map = create_occupancy_map(occupancy, grid_rows, grid_cols)

    # Display frames and results
    det_out.image(detection_vis, channels="BGR", use_container_width=True)
    occ_out.image(occ_map, use_container_width=True)
    det_boxes.write(f"Detected boxes: {len(results[0].boxes.cls)} | Missing cells: {64 - len(results[0].boxes.cls)}")

    summary = {
        "Total Spaces": len(spaces),
        "Initial": sum(1 for s in occupancy.values() if s == "initial"),
        "Empty": sum(1 for s in occupancy.values() if s == "empty"),
        "Black": sum(1 for s in occupancy.values() if s == "black"),
        "White": sum(1 for s in occupancy.values() if s == "white"),
    }
    summary_out.write(summary)
    board_status.text("\n".join(str(row) for row in new_board_status))

# Live webcam feed
def live_camera_feed():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Unable to access the camera.")
        return
    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture frame.")
        # Display the live video frame
        stframe.image(frame, channels="BGR", use_container_width=True)


def fr_capture():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Unable to access the camera.")
        return

    ret, frame = cap.read()
    if not ret:
        st.error("Failed to capture frame.")
    else:
        # Display the live video frame
        process_frame(frame)


if capture.button("Capture"):
    fr_capture()

live_camera_feed()

