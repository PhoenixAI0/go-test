import types

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from go_rl.board import BoardState
from go_rl.model_interface import TextGoPolicy


class CharacterTokenizer:
    eos_token_id = 2
    pad_token_id = 0
    vocabulary = {"A": 3, "B": 4, "1": 5, "2": 6, "P": 7, "S": 8, "X": 9}

    def encode(self, text, add_special_tokens=False):
        return [self.vocabulary[character] for character in text]

    def decode(self, ids, skip_special_tokens=True):
        inverse = {token: character for character, token in self.vocabulary.items()}
        return "".join(inverse[token] for token in ids if token in inverse)

    def apply_chat_template(self, *args, **kwargs):
        return "P"

    def __call__(self, text, **kwargs):
        return transformers.BatchEncoding({
            "input_ids": torch.tensor([[7]]),
            "attention_mask": torch.ones(1, 1, dtype=torch.long),
        })


def _probability_model(transitions):
    """Real HF generate/cache/beam machinery with known next-token probabilities."""
    model = transformers.Qwen3ForCausalLM(transformers.Qwen3Config(
        vocab_size=16, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=2, head_dim=8,
        eos_token_id=2, pad_token_id=0,
    )).eval()
    original_forward = model.forward
    original_generate = model.generate
    observed = {}

    def forward(self, input_ids=None, **kwargs):
        output = original_forward(input_ids=input_ids, **kwargs)
        logits = torch.full_like(output.logits, -100.)
        for row, last_token in enumerate(input_ids[:, -1].tolist()):
            probabilities = transitions.get(last_token, {2: 1.})
            for token, probability in probabilities.items():
                logits[row, -1, token] = torch.tensor(probability).log()
        output.logits = logits
        return output

    def generate(self, *args, **kwargs):
        observed.update(kwargs)
        return original_generate(*args, **kwargs)

    model.forward = types.MethodType(forward, model)
    model.generate = types.MethodType(generate, model)
    return model, observed


@pytest.mark.parametrize("transitions,expected", [
    # Prefix A is more likely, but its mass is split: P(A1)=.306 < P(B1)=.396.
    ({7: {3: .6, 4: .4}, 3: {5: .51, 6: .49}, 4: {5: .99, 6: .01}}, "B1"),
    # EOS likelihood matters: forced legal EOS must retain its original .01
    # probability after A1, not become probability 1 after local renormalization.
    ({7: {3: .6, 4: .4}, 3: {5: .99, 6: .01}, 4: {5: .01, 6: .99},
      5: {2: .01, 9: .99}, 6: {2: .99, 9: .01}}, "B2"),
])
def test_real_hf_beam_recovers_complete_move_probability_including_eos(transitions, expected):
    model, observed = _probability_model(transitions)
    tokenizer, board = CharacterTokenizer(), BoardState.empty(2)
    greedy = TextGoPolicy(model, tokenizer, max_new_tokens=5).predict(board)
    assert greedy.move == "A1" and greedy.legal and not greedy.truncated
    # A saved checkpoint may contain generation preferences unrelated to Go.
    # They must not silently change the advertised complete-string objective.
    model.generation_config.suppress_tokens = [4]
    model.generation_config.repetition_penalty = 2.
    for width in (2, 4, 8):
        beam = TextGoPolicy(model, tokenizer, max_new_tokens=5, num_beams=width).predict(board)
        assert beam.move == expected and beam.legal and not beam.truncated
        assert observed["renormalize_logits"] is False
        assert observed["length_penalty"] == 0.
        assert observed["early_stopping"] == "never"
        assert observed["do_sample"] is False
        assert observed["num_return_sequences"] == 1
        assert observed["generation_config"].suppress_tokens is None
        assert observed["generation_config"].repetition_penalty == 1.
        assert observed["use_model_defaults"] is False


@pytest.mark.parametrize("options", [
    {"num_beams": 0}, {"num_beams": 1.5}, {"num_beams": True},
    {"num_beams": 4, "temperature": .8},
    {"num_beams": 4, "constrained": False},
    {"num_beams": 4, "thinking": True},
])
def test_beam_rejects_ambiguous_or_unsupported_policies(options):
    with pytest.raises(ValueError):
        TextGoPolicy(None, None, **options)
