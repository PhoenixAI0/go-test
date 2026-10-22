import copy

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from go_rl.text_distill import collate_completions, completion_loss, completion_losses, uniform_objective_metadata


def _model():
    torch.manual_seed(17)
    return Qwen3ForCausalLM(Qwen3Config(
        vocab_size=37, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        pad_token_id=0, attention_dropout=0.0,
    ))


RECORDS = [
    {"prompt_ids": [1, 2, 3, 4], "answer_ids": [8, 9, 2]},
    {"prompt_ids": [5], "answer_ids": [10, 2]},
    {"prompt_ids": [3, 4], "answer_ids": [11, 12, 13, 2]},
]


def _full_lm_batch(records):
    width = max(len(row["prompt_ids"]) + len(row["answer_ids"]) for row in records)
    ids = torch.zeros((len(records), width), dtype=torch.long)
    attention = torch.zeros_like(ids)
    labels = torch.full_like(ids, -100)
    for index, row in enumerate(records):
        sequence = row["prompt_ids"] + row["answer_ids"]
        ids[index, -len(sequence):] = torch.tensor(sequence)
        attention[index, -len(sequence):] = 1
        labels[index, -len(row["answer_ids"]):] = torch.tensor(row["answer_ids"])
    positions = attention.cumsum(-1) - 1
    positions.masked_fill_(attention == 0, 0)
    return dict(input_ids=ids, attention_mask=attention, position_ids=positions, labels=labels,
                use_cache=False)


def test_trailing_projection_matches_standard_full_logits_loss_and_all_gradients():
    optimized = _model().train()
    full = copy.deepcopy(optimized)
    batch = collate_completions(RECORDS, 0)
    fast_loss = completion_loss(optimized, batch)
    reference_loss = full(**_full_lm_batch(RECORDS)).loss
    torch.testing.assert_close(fast_loss, reference_loss, atol=1e-6, rtol=1e-6)
    fast_loss.backward()
    reference_loss.backward()
    for (name, parameter), (other_name, other) in zip(optimized.named_parameters(), full.named_parameters()):
        assert name == other_name
        assert parameter.grad is not None and other.grad is not None
        torch.testing.assert_close(parameter.grad, other.grad, atol=2e-6, rtol=2e-5, msg=name)


def test_variable_length_batch_loss_matches_individual_completions():
    model = _model().eval()
    batch = collate_completions(RECORDS, 0)
    batched, mask = completion_losses(model, batch)
    assert batch.targets.tolist() == [[-100, 8, 9, 2], [-100, -100, 10, 2], [11, 12, 13, 2]]
    for index, record in enumerate(RECORDS):
        single, _ = completion_losses(model, collate_completions([record], 0))
        torch.testing.assert_close(batched[index][mask[index]], single[0], atol=1e-6, rtol=1e-6)
    reference, reference_mask = completion_losses(model, batch, efficient=False)
    torch.testing.assert_close(reference, batched)
    assert torch.equal(mask, reference_mask)


def test_completion_batches_reject_empty_context_or_targets():
    with pytest.raises(ValueError, match="nonempty"):
        collate_completions([], 0)
    with pytest.raises(ValueError, match="nonempty"):
        collate_completions([{"prompt_ids": [], "answer_ids": [1]}], 0)


def test_saved_text_context_normalizes_equivalent_rules_and_rejects_mixed_objectives():
    from go_rl.board import BoardState
    from go_rl.data_prep import state_record_from_board
    from go_rl.rules import normalize_rules
    from go_rl.teacher_data import TARGET_RULES

    row = {**state_record_from_board(BoardState.empty(19)), "rules": TARGET_RULES, "komi": 7.5}
    equivalent = {**row, "rules": normalize_rules(TARGET_RULES)}
    assert uniform_objective_metadata(iter([row, equivalent])) == {
        "board_size": 19, "rules": normalize_rules(TARGET_RULES), "komi": 7.5}
    changed_board = {**row, **state_record_from_board(BoardState.empty(9))}
    for changed in ({**row, "rules": "chinese"}, {**row, "komi": 6.5}, changed_board):
        with pytest.raises(ValueError, match="uniform"):
            uniform_objective_metadata([row, changed])
    with pytest.raises(ValueError, match="finite"):
        uniform_objective_metadata([{**row, "komi": float("nan")}])
