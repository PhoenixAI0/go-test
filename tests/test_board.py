import pytest

from go_rl.board import BoardState, GtpParseError, coords_to_gtp, gtp_to_coords


def test_gtp_roundtrip_and_pass():
    coords = gtp_to_coords("A1", board_size=5)
    assert coords == (4, 0)
    assert coords_to_gtp(coords, board_size=5) == "A1"
    assert gtp_to_coords("PASS", board_size=5) is None
    assert coords_to_gtp(None, board_size=5) == "PASS"


def test_gtp_invalid_inputs():
    with pytest.raises(GtpParseError):
        gtp_to_coords("", board_size=5)
    with pytest.raises(GtpParseError):
        gtp_to_coords("I3", board_size=5)
    with pytest.raises(GtpParseError):
        gtp_to_coords("A0", board_size=5)
    with pytest.raises(GtpParseError):
        gtp_to_coords("A9", board_size=5)


def test_board_state_legality_and_apply():
    board = BoardState.empty(5)
    assert board.is_basic_legal(None)
    assert board.is_basic_legal((4, 0))
    board.apply_move("B", "A1")
    assert (4, 0) in board.occupied()
    assert not board.is_basic_legal((4, 0))
    assert not board.is_basic_legal((5, 0))
    with pytest.raises(ValueError):
        board.apply_move("W", "A1")


def test_strict_legality_detects_suicide():
    board = BoardState.empty(3)
    board.apply_move("W", "A2")
    board.apply_move("W", "B1")
    board.apply_move("W", "C2")
    board.apply_move("W", "B3")
    coords = gtp_to_coords("B2", board_size=3)
    assert coords is not None
    assert not board.is_strict_legal("B", coords)


def test_strict_legality_detects_ko():
    board = BoardState.empty(5)
    board.apply_move("B", "B3")
    board.apply_move("B", "D3")
    board.apply_move("B", "C4")
    board.apply_move("W", "C3")
    board.apply_move("W", "B2")
    board.apply_move("W", "C1")
    board.apply_move("W", "D2")
    coords_capture = gtp_to_coords("C2", board_size=5)
    assert coords_capture is not None
    assert board.is_strict_legal("B", coords_capture)
    board_after = BoardState(size=5, moves=list(board.moves))
    board_after.apply_move("B", "C2")
    coords_recap = gtp_to_coords("C3", board_size=5)
    assert coords_recap is not None
    assert not board_after.is_strict_legal("W", coords_recap)


def test_basic_legality_allows_recapture_after_capture():
    board = BoardState.empty(3)
    board.apply_move("B", "B2")
    board.apply_move("W", "A2")
    board.apply_move("W", "B1")
    board.apply_move("W", "C2")
    board.apply_move("W", "B3")
    coords = gtp_to_coords("B2", board_size=3)
    assert coords is not None
    assert coords not in board.occupied()
    assert board.is_basic_legal(coords)
    board.apply_move("B", "B2")


def test_next_player_infers_from_history():
    board = BoardState.empty(5)
    assert board.next_player() == "B"
    board.apply_move("B", "A1")
    assert board.next_player() == "W"
    board.apply_move("W", "B1")
    assert board.next_player() == "B"
