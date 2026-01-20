from go_rl.sgf_utils import SgfGame, sample_positions


def test_sample_positions_outputs_prompts_and_targets():
    game = SgfGame(
        board_size=5,
        moves=[("B", (4, 0)), ("W", None), ("B", (3, 1))],
    )
    samples = sample_positions(game, max_positions=2)
    assert len(samples) == 2
    assert samples[0]["prompt"] == "board_size: 5\nmove:"
    assert samples[0]["target"] == "A1"
    assert samples[1]["prompt"].startswith("board_size: 5\nB: A1")
    assert samples[1]["target"] == "PASS"
