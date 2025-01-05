def render_board(board):
    import chess.svg
    return chess.svg.board(board=board)


def update_board_display(board, board_svg_placeholder):
    import base64
    board_svg = render_board(board)
    encoded_svg = base64.b64encode(board_svg.encode('utf-8')).decode('utf-8')
    board_svg_placeholder.markdown(f'<img src="data:image/svg+xml;base64,{encoded_svg}" width="400"/>', unsafe_allow_html=True)


def detect_move(previous_board_status, new_board_status, chessboard):
    move = {}
    for row in range(len(previous_board_status)):
        for col in range(len(previous_board_status[row])):
            if previous_board_status[row][col] != new_board_status[row][col]:
                if new_board_status[row][col] == 'empty' and previous_board_status[row][col] != 'empty':
                    move['start'] = (row, col)
                    move['piece'] = chessboard[row][col]
                elif previous_board_status[row][col] == 'empty' and new_board_status[row][col] != 'empty':
                    move['end'] = (row, col)
                elif previous_board_status[row][col] != new_board_status[row][col]:
                    move['end'] = (row, col)
                    move['eliminated'] = chessboard[row][col]
    return move


def update_chessboard(move, chessboard):
    start = move['start']
    end = move['end']
    piece = move['piece']
    chessboard[start[0]][start[1]] = '.'
    chessboard[end[0]][end[1]] = piece
    return chessboard


def check_win_condition(board, winner_announcement):
    if board.is_checkmate():
        winner = "Black" if board.turn else "White"
        winner_announcement.success(f"Checkmate! {winner} wins the game!")
        return True
    elif board.is_stalemate():
        winner_announcement.warning("The game is a draw due to stalemate!")
        return True
    elif board.is_insufficient_material():
        winner_announcement.warning("The game is a draw due to insufficient material!")
        return True
    elif board.is_seventyfive_moves():
        winner_announcement.warning("The game is a draw due to the 75-move rule!")
        return True
    elif board.is_fivefold_repetition():
        winner_announcement.warning("The game is a draw due to fivefold repetition!")
        return True
    return False


def export_to_pdf(st, canvas, white_moves, black_moves):
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
    st.download_button(
        label="Download Move History as PDF",
        data=pdf,
        file_name="chess_move_history.pdf",
        mime="application/pdf"
    )
