"""Exercise actual Transformers/PEFT checkpoint persistence without a download."""
import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3Model

from go_rl.policy_model import create_policy_model, load_policy_model, save_policy_model


@pytest.mark.parametrize("adapter", [False, True])
@pytest.mark.parametrize("readout,attention", [("last", "causal"), ("cells", "bidirectional"), ("cell_attention", "causal")])
def test_policy_checkpoint_restores_trained_outputs(tmp_path, adapter, readout, attention):
    torch.manual_seed(73)
    base_path = tmp_path / "base"
    base = Qwen3Model(Qwen3Config(
        vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        pad_token_id=0, attention_dropout=0.0,
    ))
    base.save_pretrained(base_path)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({"[PAD]": 0, "[EOS]": 1, "[UNK]": 2}, unk_token="[UNK]")),
        pad_token="[PAD]", eos_token="[EOS]", unk_token="[UNK]",
    )
    tokenizer.save_pretrained(base_path)
    model, tokenizer, metadata = create_policy_model(
        str(base_path), board_size=3, device="cpu", train_backbone=adapter,
        lora_rank=2, lora_alpha=4, readout=readout, attention=attention,
    )
    inputs = {"input_ids": torch.tensor([[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]]),
              "attention_mask": torch.ones((1, 12), dtype=torch.long)}
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.01)
    for _ in range(2):
        optimizer.zero_grad()
        outputs = model(**inputs)
        (outputs.logits.square().mean() - outputs.logits[0, 4] + (outputs.value - .5).square().mean()).backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        expected = model(**inputs)
    checkpoint = tmp_path / "checkpoint"
    save_policy_model(model, tokenizer, checkpoint, metadata)
    restored, _, loaded_metadata = load_policy_model(checkpoint, "cpu")
    with torch.no_grad():
        actual = restored(**inputs)
    torch.testing.assert_close(actual.logits, expected.logits)
    torch.testing.assert_close(actual.value, expected.value)
    assert loaded_metadata == metadata
    assert not restored.training
    assert not any(parameter.requires_grad for parameter in restored.backbone.parameters())
