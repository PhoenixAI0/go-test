"""A deterministic final_score regression, separate from neural score estimates."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shlex
import shutil

import pytest
from sgfmill import sgf

from go_rl.board import BoardState
from go_rl.gtp_match import GtpEngine
from go_rl.rules import normalize_rules
from go_rl.scoring import count_board_area, terminal_board_area_result


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/benson_raw_area_19x19.sgf"
RULES = "koPOSITIONALscoreAREAtaxNONEsui0"
ROWS = [".OX.X.X............", "OXXXXXX............", "XX................."] + ["." * 19] * 16


def _fixture_board() -> BoardState:
    game = sgf.Sgf_game.from_bytes(FIXTURE.read_bytes())
    root = game.get_root()
    assert game.get_size() == 19 and root.get("KM") == 7.5
    assert normalize_rules(root.get("RU")) == normalize_rules(RULES)
    matrix = [[0] * 19 for _ in range(19)]
    black, white, empty = root.get_setup_stones()
    assert not empty and not (black & white)
    for value, stones in ((1, black), (2, white)):
        for row, column in stones:
            matrix[18 - row][column] = value
    assert matrix == [[".XO".index(point) for point in row] for row in ROWS]
    board = BoardState.from_matrix(19, matrix, root.get("PL").upper())
    for node in game.get_main_sequence()[1:]:
        player, coords = node.get_move()
        assert coords is None and player is not None
        assert player.upper() == board.next_player()
        board.apply_move(player, "PASS", strict=True)
    return board


def test_benson_fixture_raw_area_retains_present_white_corner_stones():
    board = _fixture_board()
    area = count_board_area(board.current_board())
    assert (area.black, area.white, area.neutral) == (358, 3, 0)
    assert board.consecutive_passes() == 2
    assert terminal_board_area_result(board, komi=7.5, rules=RULES) == "B+347.5"


@pytest.mark.katago
def test_real_katago_benson_final_score_differs_from_raw_board_area(tmp_path):
    """Use loadsgf/final_score only; no genmove, analysis, or scoreLead fallback."""
    binary = os.environ.get("KATAGO_BIN")
    model = os.environ.get("KATAGO_MODEL")
    config = Path(os.environ.get("KATAGO_GTP_CONFIG", ROOT / "config/katago_gtp_eval.cfg"))
    if not binary or not model:
        pytest.skip("Set KATAGO_BIN and KATAGO_MODEL for the real GTP final_score regression")
    binary = shutil.which(binary) or binary
    if not Path(binary).is_file() or not Path(model).is_file() or not config.is_file():
        pytest.skip("KataGo binary, network, and separate GTP config must exist")
    argv = [binary, "gtp", "-model", model, "-config", str(config),
            "-override-config", "maxVisits=1,numSearchThreads=1"]
    with (tmp_path / "katago_gtp.log").open("w") as log:
        engine = GtpEngine("KataGo scoring regression", shlex.join(argv), stderr=log)
        try:
            version = engine.send("version")
            assert re.search(r"\b1\.18\.2\b", version), f"Fixture pins KataGo 1.18.2, got {version!r}"
            engine.send(f"loadsgf {FIXTURE}")
            loaded_rules = json.loads(engine.send("kata-get-rules"))
            assert normalize_rules(loaded_rules) == normalize_rules(RULES)
            # kata-get-rules deliberately excludes komi in v1.18.2.
            assert float(engine.send("get_komi")) == 7.5
            # Two passes in the loaded SGF finish the game, so final_score uses
            # the deterministic pass-alive scorer rather than running a search.
            result = engine.send("final_score")
            repeated = engine.send("final_score")
        finally:
            engine.close()
    raw = terminal_board_area_result(_fixture_board(), komi=7.5, rules=RULES)
    assert raw == "B+347.5"
    assert result == repeated == "B+353.5"
    assert float(result[2:]) - float(raw[2:]) == 6.0
