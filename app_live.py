import streamlit as st
import cv2
from ultralytics import YOLO
from helpers import *

# Load your YOLOv8 model
model = YOLO('weights/bestV5.pt')

st.title("Chess Detection with Occupancy Grid")

# Sidebar settings
st.sidebar.title("Settings")
conf_threshold = st.sidebar.slider("Confidence threshold", min_value=0.0, max_value=1.0, value=0.05, step=0.05)
grid_rows = grid_cols = 8


stframe = st.empty() 
col1, col2 = st.columns(2)
with col1:
    det_out = st.empty()
with col2:
    occ_out = st.empty()
det_boxes = st.empty()
summary_out = st.empty()
board_status = st.empty()

# Process the frame and create occupancy map
def process_frame(frame):
    spaces = create_grid(grid_rows, grid_cols)
    results = model.predict(source=frame, conf=conf_threshold)

    new_shape, lm, bm = get_frame_new_shape(results[0].boxes.xyxy, frame.shape)

    # Perform YOLO prediction
    boxes = results[0].boxes.xyxy.cpu().numpy() if results else []


    predicted_classes = results[0].boxes.cls
    class_names = model.names
    predicted_class_names = [class_names[int(cls_idx)] for cls_idx in predicted_classes]    

    occupancy = map_detections_to_spaces(boxes, spaces, predicted_class_names, new_shape, grid_rows, grid_cols, lm, bm)

    new_board_status = map_occupancy_to_board_status(occupancy)


    # Visualize detections
    detection_vis = results[0].plot() if results else frame

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

    return occ_map
# Live webcam feed
def live_camera_feed():
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        st.error("Unable to access the camera.")
        return

    stop_button = st.button("Stop Camera Feed")  
    capture_button = st.button("Capture Frame")  # Button to manually capture a frame

    frame_counter = 0  

    while True:
        ret, frame = cap.read()

        if not ret:
            st.error("Failed to capture frame.")
            break
        
        # Display the live video frame (to be deleted)
        stframe.image(frame, channels="BGR", use_container_width=True)

        # # plan A : derease frame rate to 15 fps to to avoid noise in the detections, but it will increase processing time overhead
        # frame_counter += 1
        # if frame_counter % 15 == 0:
        #         occ_map = process_frame(frame)
                
        #         # plan C : procces frame after evere 15 fbs  and check if procces frame returns 'initial' status if so, increse the frame counter by 5 and start the process_frame again
        #         if any(s == 'initial' for s in occ_map.values()):
        #             frame_counter += 4

        # plan  B : add button to captuer frame and pass it to process_frame (timer function in real match)
        if capture_button:
           process_frame(frame)


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
