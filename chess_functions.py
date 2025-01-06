import streamlit as st
import chess
import chess.svg
import pandas as pd
import base64
from reportlab.pdfgen import canvas
import chess.engine

# Variables
piece_names = {
    'wr': 'Rook', 'wn': 'Knight', 'wb': 'Bishop', 'wq': 'Queen', 'wk': 'King', 'wp': 'Pawn',
    'br': 'Rook', 'bn': 'Knight', 'bb': 'Bishop', 'bq': 'Queen', 'bk': 'King', 'bp': 'Pawn'
}

def start_game(chessboard = None):
    st.session_state.board = chess.Board()
    st.session_state.move_history = []
    st.session_state.chessboard = [
        ['br', 'bn', 'bb', 'bq', 'bk', 'bb', 'bn', 'br'],
        ['bp', 'bp', 'bp', 'bp', 'bp', 'bp', 'bp', 'bp'],
        ['.', '.', '.', '.', '.', '.', '.', '.'],
        ['.', '.', '.', '.', '.', '.', '.', '.'],
        ['.', '.', '.', '.', '.', '.', '.', '.'],
        ['.', '.', '.', '.', '.', '.', '.', '.'],
        ['wp', 'wp', 'wp', 'wp', 'wp', 'wp', 'wp', 'wp'],
        ['wr', 'wn', 'wb', 'wq', 'wk', 'wb', 'wn', 'wr']
    ]
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
    st.session_state.white_moves = pd.DataFrame(columns=["Piece", "From", "To", "Eliminated"])
    st.session_state.black_moves = pd.DataFrame(columns=["Piece", "From", "To", "Eliminated"])

def render_board(board):
    return chess.svg.board(board=board)

def update_board_display(board):
    board_svg = render_board(board)
    encoded_svg = base64.b64encode(board_svg.encode('utf-8')).decode('utf-8')
    return f'<img src="data:image/svg+xml;base64,{encoded_svg}" width="400"/>'

def detect_move(previous_board_status, new_board_status, chessboard, board: chess.Board):
    move = {}
    for row in range(len(previous_board_status)):
        for col in range(len(previous_board_status[row])):
            if previous_board_status[row][col] != new_board_status[row][col]:
                if new_board_status[row][col] == 'empty' and previous_board_status[row][col] != 'empty':
                    move['start'] = (row, col)
                    move['piece'] = chessboard[row][col]
                elif previous_board_status[row][col] == 'empty' and new_board_status[row][col] != 'empty':
                    move['end'] = (row, col)
                    move['captured'] = chessboard[row][col] if chessboard[row][col] != 'empty' else None

    # Suggest move only if just start is detected
    if 'start' in move and 'end' in move:
        move['end'] = suggest_move(move, board)
        return move

    return None


def suggest_full_move(board: chess.Board):
    engine_path = "stockfish"
    engine = chess.engine.SimpleEngine.popen_uci(engine_path)
    
    try:
        # Get the best move from Stockfish
        result = engine.play(board, chess.engine.Limit(time=0.5))
        best_move = result.move

        # Convert the move to row-column format
        start_square = best_move.from_square
        end_square = best_move.to_square
        
        # Check for any eliminated piece
        eliminated_piece = board.piece_at(end_square)  # Get the piece at the destination square
        eliminated_piece_str = (
            eliminated_piece.symbol() if eliminated_piece else None
        )  # Symbol of the eliminated piece (e.g., 'p', 'N')

        if eliminated_piece_str:
            if eliminated_piece_str.isupper():  # Uppercase indicates a white piece
                eliminated_piece_str = 'w' + eliminated_piece_str.lower()
            else:  # Lowercase indicates a black piece
                eliminated_piece_str = 'b' + eliminated_piece_str.lower()


        # Construct the move dictionary
        move = {
            'start': (7 - chess.square_rank(start_square), chess.square_file(start_square)),
            'end': (7 - chess.square_rank(end_square), chess.square_file(end_square)),
            'is_suggested': True,
            'eliminated': eliminated_piece_str,
        }
        return move
    finally:
        engine.quit()



def suggest_move(move, board: chess.Board):
    start_square = chess.square(move['start'][1], 7 - move['start'][0])
    legal_moves = [m for m in board.legal_moves if m.from_square == start_square]

    if legal_moves:
        # Choose the best move for the piece (heuristic or Stockfish evaluation)
        suggested_move = legal_moves[0]
        end_square = suggested_move.to_square
        return (7 - chess.square_rank(end_square), chess.square_file(end_square))

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
