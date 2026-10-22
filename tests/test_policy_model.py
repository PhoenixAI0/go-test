import torch
import pytest
from transformers import Qwen3Config, Qwen3Model

from go_rl.policy_model import GoPolicyValueModel, PolicyValueOutput, policy_value_loss


def tiny_policy(**kwargs):
    torch.manual_seed(17)
    return GoPolicyValueModel(Qwen3Model(Qwen3Config(
        vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        pad_token_id=0, attention_dropout=0.0,
    )), board_size=3, **kwargs)


def test_padding_does_not_change_policy_or_value():
    model = tiny_policy().eval()
    unpadded = model(torch.tensor([[1, 2, 3]]), torch.tensor([[1, 1, 1]]))
    right = model(torch.tensor([[1, 2, 3, 0, 0]]), torch.tensor([[1, 1, 1, 0, 0]]))
    left = model(torch.tensor([[0, 0, 1, 2, 3]]), torch.tensor([[0, 0, 1, 1, 1]]))
    torch.testing.assert_close(unpadded.logits, right.logits, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(unpadded.logits, left.logits, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(unpadded.value, right.value)


def test_soft_targets_with_illegal_actions_have_finite_gradients():
    logits = torch.tensor([[0.0, 1.0, float("-inf")]], requires_grad=True)
    value = torch.tensor([0.0], requires_grad=True)
    loss, _ = policy_value_loss(PolicyValueOutput(logits, value), torch.tensor([[.2, .8, 0.]]),
                                value_targets=torch.tensor([1.]))
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(logits.grad).all() and logits.grad[0, 2] == 0
    assert value.grad.item() < 0  # Gradient descent raises a winning value.
    with pytest.raises(ValueError, match="illegal"):
        policy_value_loss(PolicyValueOutput(logits, value), torch.tensor([[.2, .7, .1]]))


def test_real_transformer_update_learns_target_and_keeps_action_mask():
    model = tiny_policy()
    ids, mask = torch.tensor([[1, 2, 3]]), torch.ones((1, 3), dtype=torch.long)
    legal = torch.tensor([[True, False, True, False, True, False, True, False, True, True]])
    target = torch.zeros((1, 10))
    target[0, 4] = 1
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    before = model(ids, mask, legal).logits.softmax(-1)[0, 4].item()
    for _ in range(8):
        out = model(ids, mask, legal)
        loss, _ = policy_value_loss(out, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    after = model(ids, mask, legal).logits.softmax(-1)
    assert after[0, 4].item() > before + .25
    assert after[~legal].sum().item() == 0


def test_reject_empty_observations_and_action_masks():
    model = tiny_policy()
    with pytest.raises(ValueError, match="unmasked"):
        model(torch.tensor([[1]]), torch.tensor([[0]]))
    with pytest.raises(ValueError, match="at least one action"):
        model(torch.tensor([[1]]), torch.tensor([[1]]), torch.zeros((1, 10), dtype=torch.bool))


@pytest.mark.parametrize("attention", ["causal", "bidirectional"])
@pytest.mark.parametrize("readout", ["cells", "cell_attention"])
def test_spatial_readout_preserves_padding_and_future_context(attention, readout):
    model = tiny_policy(readout=readout, attention=attention, cell_suffix_tokens=1).eval()
    # Header, nine distinct cells, suffix. The final cell has a unique token.
    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 19, 20]])
    mask = torch.ones_like(ids)
    expected = model(ids, mask)
    assert expected.logits.shape == (1, 10)
    for left in (False, True):
        padding = torch.zeros((1, 2), dtype=torch.long)
        padded = torch.cat((padding, ids) if left else (ids, padding), 1)
        padded_mask = torch.cat((padding, mask) if left else (mask, padding), 1)
        actual = model(padded, padded_mask)
        torch.testing.assert_close(actual.logits, expected.logits, atol=1e-6, rtol=1e-5)
    expected.logits[0, 0].backward()
    late_cell_gradient = model.backbone.get_input_embeddings().weight.grad[19].abs().sum().item()
    assert (late_cell_gradient > 0) == (attention == "bidirectional" or readout == "cell_attention")
