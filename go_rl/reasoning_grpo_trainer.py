from __future__ import annotations

from collections import defaultdict

import torch
from trl import GRPOTrainer

from .config import ReasoningConfig
from .reasoning import (
    build_chat_template_kwargs,
    build_final_only_mask,
    build_generation_kwargs,
    detect_reasoning_profile,
)


class ReasoningGRPOTrainer(GRPOTrainer):
    def __init__(
        self,
        *args: object,
        reasoning_config: ReasoningConfig | None = None,
        reasoning_model_name: str = "",
        **kwargs: object,
    ) -> None:
        self.reasoning_config = reasoning_config or ReasoningConfig()
        super().__init__(*args, **kwargs)
        self._restore_checkpointing_options()

        model_name = reasoning_model_name or getattr(getattr(self.model, "config", None), "_name_or_path", "")
        self.reasoning_profile = detect_reasoning_profile(
            model_name=model_name,
            tokenizer=self.processing_class,
            configured_profile=self.reasoning_config.profile,
        )

        chat_kwargs = build_chat_template_kwargs(self.reasoning_config, self.reasoning_profile)
        if chat_kwargs:
            merged = dict(getattr(self, "chat_template_kwargs", {}) or {})
            merged.update(chat_kwargs)
            self.chat_template_kwargs = merged
            args_kwargs = dict(getattr(self.args, "chat_template_kwargs", {}) or {})
            args_kwargs.update(chat_kwargs)
            self.args.chat_template_kwargs = args_kwargs

        generation_kwargs = build_generation_kwargs(self.reasoning_config, self.reasoning_profile)
        if generation_kwargs:
            merged_generation = dict(getattr(self.args, "generation_kwargs", {}) or {})
            merged_generation.update(generation_kwargs)
            self.args.generation_kwargs = merged_generation
            self.generation_kwargs = merged_generation

        if not hasattr(self, "_metrics"):
            self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}

    def _restore_checkpointing_options(self) -> None:
        # TRL 0.24 enables checkpointing without forwarding kwargs, both during
        # setup and after generation. Reentrant checkpointing can lose all LoRA
        # gradients when the frozen embeddings don't require gradients.
        if self.model.is_gradient_checkpointing:
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs=self.args.gradient_checkpointing_kwargs
                or {"use_reentrant": False}
            )

    def _generate_and_score_completions(
        self,
        inputs: list[dict[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        output = super()._generate_and_score_completions(inputs)
        self._restore_checkpointing_options()
        if not self.reasoning_config.enabled or not self.reasoning_config.optimize_final_only:
            return output

        completion_ids = output["completion_ids"]
        completion_mask = output["completion_mask"]
        device = completion_mask.device

        masked_rows: list[torch.Tensor] = []
        total_tokens = int(completion_mask.sum().item())
        kept_tokens = 0
        malformed_count = 0
        final_lengths: list[int] = []

        for row_index in range(completion_ids.shape[0]):
            row_ids = completion_ids[row_index]
            row_mask = completion_mask[row_index]
            valid_len = int(row_mask.sum().item())
            masked_row = torch.zeros_like(row_mask)
            if valid_len <= 0:
                masked_rows.append(masked_row)
                continue

            token_ids = row_ids[:valid_len].tolist()
            completion_text = self.processing_class.decode(token_ids, skip_special_tokens=True)
            final_only_mask, malformed, final_len = build_final_only_mask(
                completion=completion_text,
                completion_ids=token_ids,
                tokenizer=self.processing_class,
                profile=self.reasoning_profile,
                malformed_policy=self.reasoning_config.malformed_policy,
            )
            if malformed:
                malformed_count += 1

            final_lengths.append(final_len)
            if final_only_mask:
                mask_tensor = torch.tensor(final_only_mask, dtype=row_mask.dtype, device=device)
                masked_row[:valid_len] = mask_tensor
                kept_tokens += int(mask_tensor.sum().item())
            masked_rows.append(masked_row)

        output["completion_mask"] = torch.stack(masked_rows, dim=0)

        mode = "train" if self.model.training else "eval"
        masked_token_ratio = 1.0 - (kept_tokens / max(total_tokens, 1))
        final_span_mean_len = (sum(final_lengths) / len(final_lengths)) if final_lengths else 0.0
        self._metrics[mode]["reasoning/masked_token_ratio"].append(float(masked_token_ratio))
        self._metrics[mode]["reasoning/malformed_completion_count"].append(float(malformed_count))
        self._metrics[mode]["reasoning/final_span_mean_len"].append(float(final_span_mean_len))
        return output
