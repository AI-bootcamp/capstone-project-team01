import streamlit as st
import chess
import chess.svg
from io import BytesIO
import base64
from reportlab.pdfgen import canvas


# Initialize board and move history
if 'board' not in st.session_state:
    st.session_state.board = chess.Board()
    st.session_state.move_history = []

board = st.session_state.board
move_history = st.session_state.move_history

# Function to render chessboard as SVG
def render_board(board):
    return chess.svg.board(board=board)

# Streamlit UI
st.title("Chess Game")

# Display board
board_svg_placeholder = st.empty()
def update_board_display():
    board_svg = render_board(board)
    encoded_svg = base64.b64encode(board_svg.encode('utf-8')).decode('utf-8')
    board_svg_placeholder.markdown(f'<img src="data:image/svg+xml;base64,{encoded_svg}" width="400"/>', unsafe_allow_html=True)

update_board_display()

# Move input section
if not board.is_game_over():
    current_turn = "White" if board.turn == chess.WHITE else "Black"
    st.write(f"{current_turn}'s turn")

    with st.form("move_form"):
        start_square = st.text_input("Start square (e.g., e2):", key='start')
        end_square = st.text_input("End square (e.g., e4):", key='end')
        submit = st.form_submit_button("Make Move")

        if submit and start_square and end_square:
            square_map = {square: chess.parse_square(square) for square in chess.SQUARE_NAMES}
            try:
                start = square_map[start_square]
                end = square_map[end_square]
                move = chess.Move(start, end)

                if move in board.legal_moves:
                    piece = board.piece_at(start)
                    piece_name = chess.piece_name(piece.piece_type).capitalize() if piece else "Unknown"
                    move_history.append(f"{current_turn} - {piece_name} from {start_square} to {end_square}")
                    board.push(move)
                    update_board_display()
                    st.rerun()  # Force rerun to switch player turn
                else:
                    st.error("Illegal move. Try again.")
            except KeyError:
                st.error("Invalid square. Use chess notation (e.g., e2, g1). Try again.")

# Display move history
st.write("### Move History")
for i, mv in enumerate(move_history, start=1):
    st.write(f"{i}. {mv}")

# Game Over Message
if board.is_game_over():
    result = board.result()
    if result == "1-0":
        st.success("White wins!")
    elif result == "0-1":
        st.success("Black wins!")
    else:
        st.info("It's a draw!")


# Export move history to PDF
if st.button("Export Move History to PDF"):
    pdf = BytesIO()
    c = canvas.Canvas(pdf)
    
    c.setFont("Helvetica", 14)
    c.drawString(100, 800, "Chess Game - Move History")
    
    y = 780
    c.setFont("Helvetica", 12)
    for i, move in enumerate(move_history, start=1):
        y -= 20
        c.drawString(100, y, f"{i}. {move}")
        
    c.save()
    pdf.seek(0)

    st.download_button(
        label="Download Move History as PDF",
        data=pdf,
        file_name="chess_move_history.pdf",
        mime="application/pdf"
    )

