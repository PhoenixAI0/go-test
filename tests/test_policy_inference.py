import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from go_rl.board import BoardState, gtp_to_coords
from go_rl.data_prep import board_from_example, state_record_from_board, transform_teacher_example
from go_rl.model_interface import legal_action_mask
from go_rl.policy_inference import action_permutation, infer_policy_batch
from go_rl.policy_model import PolicyValueOutput, collate_policy_inputs, encode_policy_position
from go_rl.teacher_data import TARGET_RULES


class BoardTokenizer:
    pad_token_id = 9

    def apply_chat_template(self, messages, **kwargs):
        text = messages[0]["content"]
        size = int(text.split("board_size: ")[1].splitlines()[0])
        player = int(text.split("to_play: ")[1][0] == "W")
        board_lines = text.split("board:\n")[1].splitlines()[:size]
        return [size, player, *(".XO".index(char) for line in board_lines for char in line.split(": ")[1])]


class AnisotropicPolicy(torch.nn.Module):
    """Orientation-dependent analytical model, deliberately without a legal mask."""
    def __init__(self, size=19):
        super().__init__()
        self.board_size = size
        self.backbone = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=1024))
        self.register_buffer("coordinate", torch.linspace(-1, 1, size * size + 1))
        self.forward_shapes = []
        self.last_legal = None

    @property
    def device(self):
        return self.coordinate.device

    def forward(self, input_ids, attention_mask, legal_mask):
        self.forward_shapes.append(tuple(input_ids.shape))
        self.last_legal = legal_mask.clone()
        cells = input_ids[:, 2:].float()
        feature = (cells * self.coordinate[:-1]).sum(-1) / self.board_size
        logits = self.coordinate[None, :] * 1.7 + feature[:, None] * self.coordinate.square()[None, :]
        logits = logits + torch.cat((cells * .4, torch.zeros_like(feature[:, None])), dim=1)
        return PolicyValueOutput(logits, (feature + input_ids[:, 1] * .2).tanh())


def source(size=19):
    board = BoardState.empty(size)
    actions = ("A1", "B3", "C2") if size == 3 else ("A1", "J9", "T18", "PASS", "D4")
    for move in actions:
        board.apply_move(board.next_player(), move, strict=True)
    return {**state_record_from_board(board), "rules": TARGET_RULES, "komi": 7.5,
            "game_id": "unchanged", "split": "validation", "response": "PASS",
            "teacher_value": -.9, "policy_target": [1 / (size * size + 1)] * (size * size + 1)}


def numpy_transformed_actions(size, symmetry):
    actions = np.arange(size * size).reshape(size, size)
    if symmetry >= 4:
        actions = np.fliplr(actions)
    return np.rot90(actions, -(symmetry % 4)).copy().reshape(-1)


@pytest.mark.parametrize("size", [3, 19])
def test_inverse_probability_average_pass_value_and_legality(size):
    row, tokenizer = source(size), BoardTokenizer()
    model = AnisotropicPolicy(size).eval()
    original = copy.deepcopy(row)
    weights = {key: value.clone() for key, value in model.state_dict().items()}
    actual = infer_policy_batch(model, tokenizer, [row], symmetries=8)
    assert model.forward_shapes == [(8, size * size + 2)]
    expected_probabilities, expected_values = [], []
    for symmetry in range(8):
        oriented = board_from_example(transform_teacher_example(row, symmetry))
        legal = torch.tensor([legal_action_mask(oriented)], dtype=torch.bool)
        ids = encode_policy_position(oriented, tokenizer, rules=TARGET_RULES)
        output = model(**collate_policy_inputs([ids], tokenizer), legal_mask=legal)
        probability = output.logits.masked_fill(~legal, float("-inf")).softmax(-1)[0]
        restored = torch.empty_like(probability)
        restored[numpy_transformed_actions(size, symmetry)] = probability[:-1]
        restored[-1] = probability[-1]
        expected_probabilities.append(restored)
        expected_values.append(output.value[0])
    torch.testing.assert_close(actual.probabilities[0], torch.stack(expected_probabilities).mean(0))
    torch.testing.assert_close(actual.value[0], torch.stack(expected_values).mean())
    legal = torch.tensor(legal_action_mask(board_from_example(row)))
    assert torch.equal(actual.probabilities[0][~legal], torch.zeros((~legal).sum()))
    assert actual.probabilities[0, -1] > 0
    assert actual.probabilities.sum().item() == pytest.approx(1)
    assert actual.prompt_tokens == [8 * (size * size + 2)]
    assert actual.orientation_prompt_tokens == [[size * size + 2] * 8]
    assert actual.forward_sequences == 8 and actual.forward_calls == 1 and actual.forward_seconds >= 0
    assert not actual.probabilities.requires_grad and not actual.value.requires_grad
    assert row == original and not model.training
    for key, value in model.state_dict().items():
        assert torch.equal(value, weights[key])


@pytest.mark.parametrize("symmetry", range(8))
def test_d4_ensemble_is_orientation_equivariant(symmetry):
    row, tokenizer = source(), BoardTokenizer()
    model = AnisotropicPolicy().eval()
    base = infer_policy_batch(model, tokenizer, [row], symmetries=8)
    rotated = infer_policy_batch(model, tokenizer, [transform_teacher_example(row, symmetry)], symmetries=8)
    expected = torch.empty_like(base.probabilities)
    expected[:, :-1] = base.probabilities[:, numpy_transformed_actions(19, symmetry)]
    expected[:, -1] = base.probabilities[:, -1]
    torch.testing.assert_close(rotated.probabilities, expected, atol=1e-8, rtol=1e-6)
    torch.testing.assert_close(rotated.value, base.value, atol=1e-7, rtol=1e-6)


def test_single_orientation_matches_direct_inference_and_batch_order():
    records, tokenizer = [source(3), transform_teacher_example(source(3), 1)], BoardTokenizer()
    model = AnisotropicPolicy(3).eval()
    actual = infer_policy_batch(model, tokenizer, records, symmetries=1)
    boards = [board_from_example(row) for row in records]
    legal = torch.tensor([legal_action_mask(board) for board in boards])
    ids = [encode_policy_position(board, tokenizer, rules=TARGET_RULES) for board in boards]
    direct = model(**collate_policy_inputs(ids, tokenizer), legal_mask=legal)
    torch.testing.assert_close(actual.probabilities, direct.logits.masked_fill(~legal, float("-inf")).softmax(-1))
    torch.testing.assert_close(actual.value, direct.value)
    assert actual.forward_sequences == 2 and actual.prompt_tokens == [11, 11]


def test_ensemble_batch_keeps_source_positions_separate():
    records = [source(3), {**state_record_from_board(BoardState.empty(3)), "rules": TARGET_RULES, "komi": 7.5}]
    model, tokenizer = AnisotropicPolicy(3).eval(), BoardTokenizer()
    batched = infer_policy_batch(model, tokenizer, records, symmetries=8)
    assert model.forward_shapes == [(16, 11)]
    for index, row in enumerate(records):
        single = infer_policy_batch(model, tokenizer, [row], symmetries=8)
        torch.testing.assert_close(batched.probabilities[index], single.probabilities[0])
        torch.testing.assert_close(batched.value[index], single.value[0])


def test_action_permutations_are_bijections_with_pass_fixed():
    for symmetry in range(8):
        permutation = action_permutation(19, symmetry)
        assert sorted(permutation) == list(range(362)) and permutation[-1] == 361
        inverse = np.argsort(numpy_transformed_actions(19, symmetry))
        assert permutation[:-1] == tuple(inverse)


def test_transformed_ko_history_masks_match_fresh_legality_and_preserve_pass():
    matrix = [[0] * 5 for _ in range(5)]
    for color, move in (("B", "B3"), ("B", "D3"), ("B", "C4"),
                        ("W", "C3"), ("W", "B2"), ("W", "C1"), ("W", "D2")):
        row, column = gtp_to_coords(move, 5)
        matrix[row][column] = 1 if color == "B" else 2
    board = BoardState.from_matrix(5, matrix, "B")
    board.apply_move("B", "C2", strict=True)
    assert "C3" not in board.legal_moves(superko=True)
    row = {**state_record_from_board(board), "rules": TARGET_RULES, "komi": 7.5}
    model = AnisotropicPolicy(5).eval()
    actual = infer_policy_batch(model, BoardTokenizer(), [row], symmetries=8)
    for symmetry in range(8):
        transformed = board_from_example(transform_teacher_example(row, symmetry))
        fresh = torch.tensor(legal_action_mask(transformed, superko=True))
        assert torch.equal(model.last_legal[symmetry], fresh)
        assert fresh[-1]
    assert actual.probabilities[0, 12].item() == 0  # Original C3 ko recapture.
    assert actual.probabilities[0, -1].item() > 0


def test_real_tiny_qwen_batched_ensemble_matches_individual_orientation_forwards():
    from transformers import Qwen3Config, Qwen3Model
    from go_rl.policy_model import GoPolicyValueModel

    torch.manual_seed(29)
    model = GoPolicyValueModel(Qwen3Model(Qwen3Config(
        vocab_size=16, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        max_position_embeddings=128, pad_token_id=9, attention_dropout=0.0,
    )), board_size=3).eval()
    with torch.no_grad():
        model.value_head.weight.fill_(.03)
    row, tokenizer = source(3), BoardTokenizer()
    actual = infer_policy_batch(model, tokenizer, [row], symmetries=8)
    probabilities, values = [], []
    for symmetry in range(8):
        single = infer_policy_batch(model, tokenizer, [transform_teacher_example(row, symmetry)])
        restored = torch.empty(10)
        restored[numpy_transformed_actions(3, symmetry)] = single.probabilities[0, :-1]
        restored[-1] = single.probabilities[0, -1]
        probabilities.append(restored)
        values.append(single.value[0])
    torch.testing.assert_close(actual.probabilities[0], torch.stack(probabilities).mean(0), atol=1e-7, rtol=1e-6)
    torch.testing.assert_close(actual.value[0], torch.stack(values).mean(), atol=1e-7, rtol=1e-6)


def test_reject_training_mode_invalid_symmetry_size_context_and_nonfinite_outputs():
    row, tokenizer, model = source(3), BoardTokenizer(), AnisotropicPolicy(3)
    with pytest.raises(ValueError, match="eval"):
        infer_policy_batch(model, tokenizer, [row])
    model.eval()
    with pytest.raises(ValueError, match="finite"):
        infer_policy_batch(model, tokenizer, [{**row, "komi": float("nan")}])
    model.readout = "cells"
    with pytest.raises(ValueError, match="cells encoding"):
        infer_policy_batch(model, tokenizer, [row], encoding="compact")
    model.readout = "last"
    for symmetry in (0, 2, True):
        with pytest.raises(ValueError, match="symmetries"):
            infer_policy_batch(model, tokenizer, [row], symmetries=symmetry)
    with pytest.raises(ValueError, match="board size"):
        infer_policy_batch(model, tokenizer, [source()])
    model.backbone.config.max_position_embeddings = 10
    with pytest.raises(ValueError, match="context"):
        infer_policy_batch(model, tokenizer, [row], symmetries=8)
    model.backbone.config.max_position_embeddings = 1024
    model.coordinate.fill_(float("nan"))
    with pytest.raises(ValueError, match="invalid policy/value"):
        infer_policy_batch(model, tokenizer, [row])
