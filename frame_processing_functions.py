import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba

import chess

def map_board_to_board_status(board):

    board_status = [['empty' for _ in range(8)] for _ in range(8)]
    
    # Loop through all squares on the chess board
    for row in range(8):
        for col in range(8):
            square = chess.square(col, 7 - row)  # Convert row, col to square index
            piece = board.piece_at(square)  # Get the piece at this square
            
            if piece:
                board_status[row][col] = 'white' if piece.color else 'black'
                
    
    return board_status

def order_detections(boxes, classes):
    board_status = [] # Will be 2-d array that contains the classes
    detections = []

    # assuming the number of boxes is always 64
    index = 0
    for box in boxes:
        x_center = ((box[0] + box[2]) / 2)
        y_center = ((box[1] + box[3]) / 2)
        detections.append({'box': index, 'x_center': x_center, 'y_center': y_center, 'class': classes[index]})

        index += 1
        
    # Step 1: Sort detections by y_center to determine rows
    detections = sorted(detections, key=lambda d: d['y_center'])

    # Step 2: Divide detections into 8 rows
    rows = [detections[i * 8:(i + 1) * 8] for i in range(8)]

    # Step 3: Sort each row by x_center to determine columns
    for row in rows:
        sorted_row = sorted(row, key=lambda d: d['x_center'])
        board_status.append([cell['class'] for cell in sorted_row])

    return board_status

def display_board_status(board_status):
    color_mapping = {
        'black': '#000000',  # Black
        'white': '#FFFFFF',  # White
        'empty': '#DDDDDD'   # Gray for empty squares
    }

    # Convert the board to an RGBA color matrix
    color_matrix = [[to_rgba(color_mapping[cell]) for cell in row] for row in board_status]

    # Plot the heatmap
    fig, ax = plt.subplots(figsize=(8, 8))

    # Create a grid
    ax.imshow(color_matrix, extent=[0, 8, 0, 8])

    return fig
