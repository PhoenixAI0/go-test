import pytest

from go_rl.board import BoardState
from go_rl.scoring import count_board_area, terminal_board_area_result
from go_rl.teacher_data import TARGET_RULES


def test_empty_board_is_neutral_not_territory_for_both_colors():
    area = count_board_area([[0] * 19 for _ in range(19)])
    assert area.black == area.white == 0 and area.neutral == 361


@pytest.mark.parametrize("matrix", [[], [[0, 0]], [[1.0]], [[True]], [[3]]])
def test_malformed_boards_fail_explicitly(matrix):
    with pytest.raises(ValueError):
        count_board_area(matrix)


def test_single_color_border_and_mixed_border_regions():
    area = count_board_area([[1, 1, 2, 2, 2],
                             [1, 0, 1, 2, 0],
                             [1, 1, 1, 2, 2],
                             [1, 0, 0, 0, 2],
                             [1, 1, 2, 2, 2]])
    assert area.black_territory == area.white_territory == 1
    assert area.neutral == 3
    assert area.black + area.white + area.neutral == 25


def test_present_opposing_stones_are_never_removed_as_dead():
    # Topologically enclosed white still counts; no life/death oracle is hidden
    # inside the deterministic scorer. Capturing before passing is necessary.
    area = count_board_area([[1, 1, 1], [1, 2, 0], [1, 1, 1]])
    assert area.black == 7 and area.white == 1 and area.neutral == 1


def test_terminal_komi_draw_and_cap_guard():
    board = BoardState.empty(19)
    with pytest.raises(ValueError, match="two-pass"):
        terminal_board_area_result(board, komi=7.5, rules=TARGET_RULES)
    board.apply_move("B", "PASS", strict=True)
    board.apply_move("W", "PASS", strict=True)
    assert terminal_board_area_result(board, komi=7.5, rules=TARGET_RULES) == "W+7.5"
    assert terminal_board_area_result(board, komi=0, rules=TARGET_RULES) == "0"
    with pytest.raises(ValueError, match="finite"):
        terminal_board_area_result(board, komi=float("nan"), rules=TARGET_RULES)


def test_terminal_scorer_rejects_resumed_finished_history():
    board = BoardState.empty(3)
    for player, move in [("B", "PASS"), ("W", "PASS"), ("B", "A1"),
                         ("W", "PASS"), ("B", "PASS")]:
        board.apply_move(player, move, strict=True)
    with pytest.raises(ValueError, match="earlier two-pass"):
        terminal_board_area_result(board, komi=7.5, rules=TARGET_RULES)


def test_terminal_scorer_includes_a_pass_known_before_snapshot():
    matrix = [[0] * 3 for _ in range(3)]
    board = BoardState.from_matrix(3, matrix, next_player="W", previous_board=matrix)
    board.apply_move("W", "PASS", strict=True)
    assert terminal_board_area_result(board, komi=7.5, rules=TARGET_RULES) == "W+7.5"
    board.apply_move("B", "A1", strict=True)
    board.apply_move("W", "PASS", strict=True)
    board.apply_move("B", "PASS", strict=True)
    with pytest.raises(ValueError, match="earlier two-pass"):
        terminal_board_area_result(board, komi=7.5, rules=TARGET_RULES)


@pytest.mark.parametrize("rules", ["chinese", "japanese", "koPOSITIONALscoreAREAtaxSEKIsui0",
                                   "koPOSITIONALscoreAREAtaxNONEsui0button1",
                                   "koPOSITIONALscoreAREAtaxNONEsui0fpok1"])
def test_no_silent_scoring_rule_conversion(rules):
    board = BoardState.empty(3)
    board.apply_move("B", "PASS", strict=True)
    board.apply_move("W", "PASS", strict=True)
    with pytest.raises(ValueError, match="Exact board-area"):
        terminal_board_area_result(board, komi=7.5, rules=rules)


def test_d4_and_color_swap_area_invariance():
    matrix = [[1, 0, 1], [1, 0, 2], [0, 2, 2]]
    original = count_board_area(matrix)
    for _ in range(4):
        matrix = [list(row) for row in zip(*matrix[::-1])]
        assert count_board_area(matrix) == original
        assert count_board_area([row[::-1] for row in matrix]) == original
    swapped = count_board_area([[3 - value if value else 0 for value in row] for row in matrix])
    assert (swapped.black, swapped.white, swapped.neutral) == (original.white, original.black, original.neutral)
