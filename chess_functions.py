import streamlit as st
import chess
import chess.svg
import pandas as pd
import base64
from reportlab.pdfgen import canvas

# Variables
piece_names = {
    'P': 'pawn', 'N': 'knight', 'B': 'bishop', 'R': 'rook', 'Q': 'queen', 'K': 'king',
    'p': 'pawn', 'n': 'knight', 'b': 'bishop', 'r': 'rook',    'q': 'queen', 'k': 'king',
}

def start_game():
    st.session_state.board = chess.Board()
    st.session_state.move_history = []
    st.session_state.previous_board_status = [
        ['black', 'black', 'black', 'black', 'black', 'black', 'black', 'black'],
        ['black', 'black', 'black', 'black', 'black', 'black', 'black', 'black'],
        ['empty', 'empty', 'empty', 'empty', 'empty', 'empty', 'empty', 'empty'],
        ['empty', 'empty', 'empty', 'empty', 'empty', 'empty', 'empty', 'empty'],
        ['empty', 'empty', 'empty', 'empty', 'empty', 'empty', 'empty', 'empty'],
        ['empty', 'empty', 'empty', 'empty', 'empty', 'empty', 'empty', 'empty'],
        ['white', 'white', 'white', 'white', 'white', 'white', 'white', 'white'],
        ['white', 'white', 'white', 'white', 'white', 'white', 'white', 'white']
    ]
    st.session_state.white_moves = pd.DataFrame(columns=["Piece", "From", "To", "Eliminated", "castle"])
    st.session_state.black_moves = pd.DataFrame(columns=["Piece", "From", "To", "Eliminated", "castle"])


def update_board_display(board):
    board_svg = chess.svg.board(board=board)
    encoded_svg = base64.b64encode(board_svg.encode('utf-8')).decode('utf-8')
    return f'<img src="data:image/svg+xml;base64,{encoded_svg}" width="400"/>'

def detect_move(previous_board_status, new_board_status, board):
    move = {}
    starts = 0
    ends = 0
    for row in range(len(previous_board_status)):
        for col in range(len(previous_board_status[row])):
            if previous_board_status[row][col] != new_board_status[row][col]:
                square = chess.square(col, 7 - row) # Returns int
                square_name = chess.square_name(chess.square(col, 7 - row)) # Returns str 'f2' 
                if new_board_status[row][col] == 'empty' and previous_board_status[row][col] != 'empty':
                    starts += 1
                    move['start'] = square_name
                    move['piece'] = piece_names[board.piece_at(square).symbol()] # map piece symbol to its name
                elif previous_board_status[row][col] == 'empty' and new_board_status[row][col] != 'empty':
                    ends +=1
                    move['end'] = square_name
                elif previous_board_status[row][col] != new_board_status[row][col]:
                    ends +=1
                    move['end'] = square_name
                    move['eliminated'] = board.piece_at(square)

    # check for castle movement
    if 'start' in move and 'end' in move:
        if move['start'] == "e1" and move['end'] == "g1":
            move['castle'] = "white_kingside"
        elif move['start'] == "e1" and move['end'] == "c1":
            move['castle'] = "white_queenside"
        elif move['start'] == "e8" and move['end'] == "g8":
            move['castle'] = "black_kingside"
        elif move['start'] == "e8" and move['end'] == "c8":
            move['castle'] = "black_queenside"
    
    # Suggest move only if just start is detected
    if 'start' in move and not 'end' in move:
        end_move = suggest_move(move, board)
        if end_move:
            move['end'] = end_move
        else:
            move['warning'] = 'No moves available'
        move['is_suggested'] = True
    return move

def suggest_move(move, board: chess.Board):
    start_square = chess.parse_square(move['start'])
    legal_moves = [m for m in board.legal_moves if m.from_square == start_square]

    if legal_moves:
        # Choose the best move for the piece (heuristic or Stockfish evaluation)
        suggested_move = legal_moves[0]
        end_square = suggested_move.to_square
        return chess.square_name(end_square)

    return None  # No legal moves available

def update_chessboard(move, chessboard):
    start = move['start']
    end = move['end']
    piece = move['piece']
    chessboard[start[0]][start[1]] = '.'
    chessboard[end[0]][end[1]] = piece
    return chessboard


def check_win_condition(board):
    if board.is_checkmate():
        winner = "Black" if board.turn else "White"  # If it's White's turn and checkmate, Black wins and vice versa
        return "success", f"Checkmate! {winner} wins the game!"
    elif board.is_stalemate():
        return "warning", "The game is a draw due to stalemate!"
    elif board.is_insufficient_material():
        return "warning", "The game is a draw due to insufficient material!"
    elif board.is_seventyfive_moves():
        return "warning", "The game is a draw due to the 75-move rule!"
    elif board.is_fivefold_repetition():
        return "warning", "The game is a draw due to fivefold repetition!"
    return None, None

def explain_illegal_move(board, chess_move):
    if not chess_move in board.generate_legal_moves():
        if chess_move not in board.generate_pseudo_legal_moves():
            return "The move doesn't follow the rules for the piece or the board position."
        else:
            return "The move would place or leave the king in check."
    return "Unknown reason"

def export_to_pdf(white_moves, black_moves):
    from io import BytesIO
    pdf = BytesIO()
    c = canvas.Canvas(pdf)
    c.setFont("Helvetica", 16)
    c.drawString(200, 800, "Chess Game Move History")
    c.setFont("Helvetica", 14)

    y = 760
    c.drawString(100, y, "White Player Moves:")
    y -= 20
    for index, row in white_moves.iterrows():
        c.drawString(100, y, f"{row['Piece']} from {row['From']} to {row['To']} (Eliminated: {row['Eliminated']})")
        y -= 20

    y -= 40
    c.drawString(100, y, "Black Player Moves:")
    y -= 20
    for index, row in black_moves.iterrows():
        c.drawString(100, y, f"{row['Piece']} from {row['From']} to {row['To']} (Eliminated: {row['Eliminated']})")
        y -= 20

    c.save()
    pdf.seek(0)
    return pdf
