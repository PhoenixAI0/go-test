import copy
import random

import pytest
import torch
from safetensors.torch import save_file

from go_rl.training_checkpoint import restore_training_state, save_training_state


def test_resume_reproduces_sampling_dropout_optimizer_and_schedule(tmp_path):
    torch.manual_seed(20)
    model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Dropout(.3), torch.nn.Linear(4, 1))
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=.5)
    rng = random.Random(45)
    data = torch.arange(18, dtype=torch.float32).reshape(6, 3) / 18

    def update(net, opt, schedule, sample):
        batch = data[[sample.randrange(6) for _ in range(3)]]
        opt.zero_grad()
        net(batch).square().mean().backward()
        opt.step()
        schedule.step()

    for _ in range(2):
        update(model, optimizer, scheduler, rng)
    saved_weights = copy.deepcopy(model.state_dict())
    save_file(saved_weights, tmp_path / "heads.safetensors")
    contract = {"steps": 4, "dataset": "fixed", "batch_size": 3}
    save_training_state(tmp_path, optimizer, scheduler, rng, step=2, contract=contract, device="cpu")
    for _ in range(2):
        update(model, optimizer, scheduler, rng)
    restored = copy.deepcopy(model)
    restored.load_state_dict(saved_weights)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=.01)
    restored_scheduler = torch.optim.lr_scheduler.StepLR(restored_optimizer, step_size=2, gamma=.5)
    restored_rng = random.Random(999)
    assert restore_training_state(tmp_path, restored_optimizer, restored_scheduler, restored_rng,
                                  contract=contract, device="cpu") == 2
    for _ in range(2):
        update(restored, restored_optimizer, restored_scheduler, restored_rng)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value, atol=0, rtol=0)
    assert restored_scheduler.get_last_lr() == scheduler.get_last_lr()
    with pytest.raises(ValueError, match="contract changed"):
        restore_training_state(tmp_path, restored_optimizer, restored_scheduler, restored_rng,
                               contract={**contract, "dataset": "changed"}, device="cpu")
    with pytest.raises(ValueError, match="trainer contract version changed"):
        restore_training_state(tmp_path, restored_optimizer, restored_scheduler, restored_rng,
                               contract={**contract, "trainer_version": 2}, device="cpu")
    save_file(model.state_dict(), tmp_path / "heads.safetensors")
    with pytest.raises(ValueError, match="weights differ"):
        restore_training_state(tmp_path, restored_optimizer, restored_scheduler, restored_rng,
                               contract=contract, device="cpu")
