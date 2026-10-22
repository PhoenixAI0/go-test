import pytest

from go_rl.board import BoardState, GtpParseError, coords_to_gtp, gtp_to_coords


def test_gtp_roundtrip_and_pass():
    coords = gtp_to_coords("A1", board_size=5)
    assert coords == (4, 0)
    assert coords_to_gtp(coords, board_size=5) == "A1"
    assert gtp_to_coords("PASS", board_size=5) is None
    assert coords_to_gtp(None, board_size=5) == "PASS"


@pytest.mark.parametrize(
    "gtp,board_size,expected_coords",
    [
        ("A1", 9, (8, 0)),
        ("A9", 9, (0, 0)),
        ("J1", 9, (8, 8)),
        ("J9", 9, (0, 8)),
        ("D4", 9, (5, 3)),
        ("A1", 19, (18, 0)),
        ("A19", 19, (0, 0)),
        ("T1", 19, (18, 18)),
        ("T19", 19, (0, 18)),
        ("K10", 19, (9, 9)),
        ("Q16", 19, (3, 15)),
    ],
)
def test_gtp_to_coords_various_boards(gtp, board_size, expected_coords):
    """Test GTP coordinate conversion for different board sizes."""
    coords = gtp_to_coords(gtp, board_size)
    assert coords == expected_coords
    # Test roundtrip
    assert coords_to_gtp(coords, board_size) == gtp


@pytest.mark.parametrize(
    "board_size",
    [5, 9, 13, 19],
)
def test_gtp_pass_works_for_all_boards(board_size):
    """Test that PASS works for all board sizes."""
    assert gtp_to_coords("PASS", board_size) is None
    assert gtp_to_coords("pass", board_size) is None
    assert coords_to_gtp(None, board_size) == "PASS"


def test_gtp_invalid_inputs():
    with pytest.raises(GtpParseError):
        gtp_to_coords("", board_size=5)
    with pytest.raises(GtpParseError):
        gtp_to_coords("I3", board_size=5)
    with pytest.raises(GtpParseError):
        gtp_to_coords("A0", board_size=5)
    with pytest.raises(GtpParseError):
        gtp_to_coords("A9", board_size=5)


@pytest.mark.parametrize(
    "invalid_gtp,board_size",
    [
        ("Z1", 9),  # Z is beyond J on 9x9
        ("A20", 19),  # Row 20 doesn't exist on 19x19
        ("U1", 19),  # U is beyond T on 19x19
        ("I1", 9),  # I is skipped in GTP notation
        ("i1", 9),  # lowercase i is also skipped
        ("A0", 9),  # Row 0 doesn't exist
        ("@1", 9),  # Invalid character
        ("1A", 9),  # Reversed format
        ("AA", 9),  # No number
        ("11", 9),  # No letter
    ],
)
def test_gtp_invalid_inputs_parametrized(invalid_gtp, board_size):
    """Test various invalid GTP inputs."""
    with pytest.raises(GtpParseError):
        gtp_to_coords(invalid_gtp, board_size)


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


@pytest.mark.parametrize("board_size", [5, 9, 13, 19])
def test_board_state_empty_creation(board_size):
    """Test creating empty boards of various sizes."""
    board = BoardState.empty(board_size)
    assert board.size == board_size
    assert len(board.moves) == 0
    assert len(board.occupied()) == 0
    assert board.next_player() == "B"


def test_board_state_occupied_after_moves():
    """Test that occupied() correctly tracks stones."""
    board = BoardState.empty(9)
    assert len(board.occupied()) == 0

    board.apply_move("B", "D4")
    assert len(board.occupied()) == 1
    assert gtp_to_coords("D4", 9) in board.occupied()

    board.apply_move("W", "F6")
    assert len(board.occupied()) == 2
    assert gtp_to_coords("F6", 9) in board.occupied()


def test_board_state_occupied_after_capture():
    """Test that occupied() updates after captures."""
    board = BoardState.empty(5)
    # Surround a white stone
    board.apply_move("W", "B2")
    board.apply_move("B", "A2")
    board.apply_move("B", "B1")
    board.apply_move("B", "C2")
    board.apply_move("B", "B3")  # Captures W at B2

    # White stone should be removed
    assert gtp_to_coords("B2", 5) not in board.occupied()
    assert len(board.occupied()) == 4  # Only black stones remain


def test_board_state_apply_move_increments_moves():
    """Test that moves list grows with each move."""
    board = BoardState.empty(9)
    assert len(board.moves) == 0

    board.apply_move("B", "D4")
    assert len(board.moves) == 1
    assert board.moves[0] == ("B", gtp_to_coords("D4", 9))

    board.apply_move("W", "PASS")
    assert len(board.moves) == 2
    assert board.moves[1] == ("W", None)


def test_board_state_copy_moves():
    """Test that BoardState can be created from move history."""
    original = BoardState.empty(9)
    original.apply_move("B", "D4")
    original.apply_move("W", "F6")

    # Create a copy from moves
    copy = BoardState(size=9, moves=list(original.moves))
    assert copy.size == original.size
    assert copy.moves == original.moves
    assert copy.occupied() == original.occupied()


def test_board_state_apply_pass_move():
    """Test that PASS moves are handled correctly."""
    board = BoardState.empty(9)
    board.apply_move("B", "PASS")

    assert len(board.moves) == 1
    assert board.moves[0] == ("B", None)
    assert len(board.occupied()) == 0  # PASS doesn't add stones


def test_board_state_apply_lowercase_move():
    """Test that lowercase moves work."""
    board = BoardState.empty(9)
    board.apply_move("b", "d4")  # lowercase player and move

    assert len(board.moves) == 1
    assert gtp_to_coords("D4", 9) in board.occupied()


def test_board_state_is_basic_legal_occupied():
    """Test that is_basic_legal rejects occupied points."""
    board = BoardState.empty(9)
    coords = gtp_to_coords("D4", 9)

    assert board.is_basic_legal(coords)
    board.apply_move("B", "D4")
    assert not board.is_basic_legal(coords)


def test_board_state_is_basic_legal_out_of_bounds():
    """Test that is_basic_legal rejects out-of-bounds coordinates."""
    board = BoardState.empty(9)
    assert not board.is_basic_legal((9, 0))  # Row 9 doesn't exist
    assert not board.is_basic_legal((0, 9))  # Col 9 doesn't exist
    assert not board.is_basic_legal((-1, 0))  # Negative row
    assert not board.is_basic_legal((0, -1))  # Negative col


def test_strict_legality_allows_capture():
    """Test that capturing moves are legal even when surrounded."""
    board = BoardState.empty(5)
    # Set up a capture scenario: B surrounds W, then B can capture
    board.apply_move("W", "B2")
    board.apply_move("B", "A2")
    board.apply_move("B", "B1")
    board.apply_move("B", "C2")

    # B3 would capture W at B2, so it's legal even though it looks like suicide
    coords = gtp_to_coords("B3", 5)
    assert board.is_strict_legal("B", coords)


def test_board_state_apply_move_raises_on_illegal():
    """Test that apply_move raises ValueError on illegal moves."""
    board = BoardState.empty(9)
    board.apply_move("B", "D4")

    # Trying to play on occupied point should raise
    with pytest.raises(ValueError):
        board.apply_move("W", "D4")


def _ko_position():
    board = BoardState.empty(5)
    for player, move in [("B", "B3"), ("B", "D3"), ("B", "C4"),
                         ("W", "C3"), ("W", "B2"), ("W", "C1"), ("W", "D2")]:
        board.apply_move(player, move, strict=True)
    before = board.current_board()
    board.apply_move("B", "C2", strict=True)
    return board, before


def test_snapshot_preserves_history_ko_and_passes():
    original, before_capture = _ko_position()
    snapshot = BoardState.from_matrix(5, original.current_board(), "W", before_capture)
    assert "C3" not in snapshot.legal_moves()
    assert snapshot.previous_board() == before_capture
    snapshot.apply_move("W", "PASS", strict=True)
    snapshot.apply_move("B", "PASS", strict=True)
    assert snapshot.consecutive_passes() == 2
    assert "C3" in snapshot.legal_moves()
    assert "C3" not in snapshot.legal_moves(superko=True)
    assert len(snapshot.position_history()) == 4
    clone = snapshot.clone()
    assert clone.moves == snapshot.moves
    assert clone.position_history() == snapshot.position_history()
    clone.apply_move("W", "A5", strict=True)
    assert clone.consecutive_passes() == 0
    assert snapshot.consecutive_passes() == 2
    assert clone.current_board() != snapshot.current_board()


def test_snapshot_analysis_setup_recovers_capture_and_subsequent_moves():
    original, before_capture = _ko_position()
    snapshot = BoardState.from_matrix(5, original.current_board(), "W", before_capture)
    snapshot.apply_move("W", "A5", strict=True)
    initial, moves = snapshot.analysis_setup()
    assert initial == before_capture
    assert moves == [("B", gtp_to_coords("C2", 5)), ("W", gtp_to_coords("A5", 5))]
    replayed = BoardState.from_matrix(5, initial, "B")
    for player, coords in moves:
        replayed.apply_move(player, coords, strict=True)
    assert replayed.current_board() == snapshot.current_board()
    assert replayed.previous_board() == snapshot.previous_board()


def test_snapshot_analysis_setup_recovers_pass_and_rejects_inconsistent_context():
    initial = [[0] * 3 for _ in range(3)]
    snapshot = BoardState.from_matrix(3, initial, "W", initial)
    assert snapshot.analysis_setup()[1] == [("B", None)]
    assert snapshot.consecutive_passes() == 1
    assert not snapshot.has_two_pass_history()
    snapshot.apply_move("W", "PASS", strict=True)
    snapshot.apply_move("B", "A3", strict=True)
    assert snapshot.consecutive_passes() == 0
    assert snapshot.has_two_pass_history()  # Includes the identifiable pass before the snapshot.
    inconsistent = BoardState.from_matrix(3, [[1, 1, 0], [0, 0, 0], [0, 0, 0]], "W", initial)
    with pytest.raises(ValueError, match="not a legal predecessor"):
        inconsistent.analysis_setup()


def test_cached_replay_invalidates_public_history_mutations():
    board = BoardState.empty(9)
    board.apply_move("B", "D4")
    first = board.current_board()
    board.moves[0] = ("B", gtp_to_coords("E5", 9))
    assert board.current_board() != first
    board.moves.clear()
    assert not board.occupied()
    board.moves.append(("W", gtp_to_coords("A1", 9)))
    assert board.next_player() == "B"
    assert board.occupied() == {gtp_to_coords("A1", 9)}


def test_returned_board_and_history_do_not_mutate_cache():
    board = BoardState.empty(9)
    board.apply_move("B", "D4")
    board.current_board()[0][0] = 2
    board.position_history()[-1][0][0] = 2
    board.previous_board()[0][0] = 2
    assert board.current_board()[0][0] == 0
    clone = board.clone()
    clone.apply_move("W", "A9")
    assert board.current_board()[0][0] == 0
    assert clone.current_board()[0][0] == 2


@pytest.mark.parametrize("size", [9, 19])
def test_random_full_games_match_independent_sgfmill_board(size):
    import random
    from sgfmill import boards

    rng = random.Random(20261022 + size)
    board = BoardState.empty(size)
    reference = boards.Board(size)
    for ply in range(2 * size * size):
        player = board.next_player()
        legal = board.legal_moves(include_pass=False)
        if not legal:
            board.apply_move(player, None, strict=True)
            continue
        move = rng.choice(legal)
        row, col = gtp_to_coords(move, size)
        board.apply_move(player, move, strict=True)
        reference.play(size - 1 - row, col, player.lower())
        expected = [[{None: 0, "b": 1, "w": 2}[reference.get(size - 1 - r, c)]
                     for c in range(size)] for r in range(size)]
        assert board.current_board() == expected, (ply, move)
        assert len(board.position_history()) == ply + 2


@pytest.mark.parametrize("matrix", [[[0, 0], [0]], [[0, 3], [0, 0]]])
def test_invalid_snapshot_rejected(matrix):
    with pytest.raises(ValueError):
        BoardState.from_matrix(2, matrix, "B")
