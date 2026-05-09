# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from verl.utils.torch_functional import distributed_masked_mean, distributed_mean_max_min_std, masked_mean


def _worker_mean(rank: int, world_size: int, rendezvous_file: str):
    # 1) set GPU and init NCCL
    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend="nccl",
        init_method=f"file://{rendezvous_file}",
        rank=rank,
        world_size=world_size,
    )

    # each rank holds tensor [rank+1]
    local = torch.tensor([float(rank + 1)], device=f"cuda:{rank}")
    mean, gmax, gmin, gstd = distributed_mean_max_min_std(local, True, True, True)

    values = [float(i + 1) for i in range(world_size)]
    exp_mean = sum(values) / len(values)
    exp_max = max(values)
    exp_min = min(values)
    var = sum((x - exp_mean) ** 2 for x in values) / (len(values) - 1)
    exp_std = var**0.5

    # all ranks should see the same result
    assert torch.allclose(mean.cpu(), torch.tensor(exp_mean)), f"mean@{rank}"
    assert torch.allclose(gmax.cpu(), torch.tensor(exp_max)), f"max@{rank}"
    assert torch.allclose(gmin.cpu(), torch.tensor(exp_min)), f"min@{rank}"
    assert torch.allclose(gstd.cpu(), torch.tensor(exp_std)), f"std@{rank}"

    dist.destroy_process_group()


@pytest.mark.parametrize(
    "value,mask,gt",
    [
        ([1.0, 2.0, 3.0, 4.0], [1, 0, 0, 1], 2.5),
        ([1.0, 2.0, float("nan"), 4.0], [1, 0, 0, 1], 2.5),
        ([1.0, 2.0, float("nan"), 4.0], [1, 0, 1, 0], float("nan")),
    ],
)
def test_masked_mean(value, mask, gt):
    res = masked_mean(torch.tensor(value), torch.tensor(mask))
    gt = torch.tensor(gt)
    assert torch.allclose(res, gt) or (torch.isnan(res) and torch.isnan(gt))


@pytest.mark.parametrize("world_size", [2, 4])
def test_distributed_mean_max_min_std(world_size, tmp_path):
    rendezvous_file = str(tmp_path / "rdzv_mean")
    os.makedirs(os.path.dirname(rendezvous_file), exist_ok=True)

    mp.spawn(
        fn=_worker_mean,
        args=(world_size, rendezvous_file),
        nprocs=world_size,
        join=True,
    )


def _worker_mask(rank: int, world_size: int, rendezvous_file: str):
    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend="nccl",
        init_method=f"file://{rendezvous_file}",
        rank=rank,
        world_size=world_size,
    )

    # build per‐rank tensor and mask
    local_tensor = torch.tensor([rank * 2 + 1.0, rank * 2 + 2.0], device=f"cuda:{rank}")
    if rank == 0:
        mask = torch.tensor([1, 0], device=f"cuda:{rank}", dtype=torch.float32)
    else:
        mask = torch.tensor([0, 1], device=f"cuda:{rank}", dtype=torch.float32)

    gmean = distributed_masked_mean(local_tensor, mask)

    valid_values = [1.0] + [2 * i + 2.0 for i in range(1, world_size)]
    expected_mean = sum(valid_values) / len(valid_values)
    assert torch.allclose(gmean.cpu(), torch.tensor(expected_mean)), f"masked_mean@{rank}"

    dist.destroy_process_group()


@pytest.mark.parametrize("world_size", [2, 4])
def test_distributed_masked_mean(world_size, tmp_path):
    rendezvous_file = str(tmp_path / "rdzv_mask")
    os.makedirs(os.path.dirname(rendezvous_file), exist_ok=True)

    mp.spawn(
        fn=_worker_mask,
        args=(world_size, rendezvous_file),
        nprocs=world_size,
        join=True,
    )


# Tests for Adaptive Step-Decay scheduler
def test_adaptive_step_decay_halving():
    """Test that LR halves at correct decay_period intervals."""
    from torch.optim import AdamW
    from verl.utils.torch_functional import get_adaptive_step_decay_schedule

    optimizer = AdamW([torch.randn(10, requires_grad=True)], lr=1e-3)
    scheduler = get_adaptive_step_decay_schedule(
        optimizer, num_warmup_steps=10, decay_period=20, min_lr_ratio=0.1
    )

    # Warmup phase: LR should increase from 0 to 1e-3
    warmup_lrs = []
    for step in range(10):
        scheduler.step()
        warmup_lrs.append(scheduler.get_last_lr()[0])

    # LR should be increasing during warmup
    assert warmup_lrs[0] < warmup_lrs[-1], "LR should increase during warmup"
    assert warmup_lrs[-1] < 1e-3, "LR should be below initial during warmup"

    # Step 10: First step after warmup, LR = 1e-3
    scheduler.step()
    assert scheduler.get_last_lr()[0] == 1e-3, f"LR should be 1e-3 after warmup, got {scheduler.get_last_lr()[0]}"

    # Steps 11-30: LR should stay at 1e-3 (within first decay_period)
    for _ in range(20):
        scheduler.step()
    assert scheduler.get_last_lr()[0] == 5e-4, f"LR should be 5e-4 after first halving, got {scheduler.get_last_lr()[0]}"

    # Steps 31-50: LR should halve to 2.5e-4
    for _ in range(20):
        scheduler.step()
    assert scheduler.get_last_lr()[0] == 2.5e-4, f"LR should be 2.5e-4 after second halving, got {scheduler.get_last_lr()[0]}"


def test_adaptive_step_decay_min_lr_floor():
    """Test that LR doesn't go below min_lr_ratio."""
    from torch.optim import AdamW
    from verl.utils.torch_functional import get_adaptive_step_decay_schedule

    optimizer = AdamW([torch.randn(10, requires_grad=True)], lr=1e-3)
    scheduler = get_adaptive_step_decay_schedule(
        optimizer, num_warmup_steps=0, decay_period=10, min_lr_ratio=0.1
    )

    # After many halvings, LR should stay at 1e-4 (10% of 1e-3)
    for _ in range(1000):
        scheduler.step()

    min_lr = 1e-4
    actual_lr = scheduler.get_last_lr()[0]
    assert actual_lr >= min_lr, f"LR should not go below {min_lr}, got {actual_lr}"
    assert actual_lr == min_lr, f"LR should stabilize at {min_lr}, got {actual_lr}"


def test_adaptive_step_decay_uninitialized():
    """Test that scheduler stays constant when decay_period=-1."""
    from torch.optim import AdamW
    from verl.utils.torch_functional import get_adaptive_step_decay_schedule

    optimizer = AdamW([torch.randn(10, requires_grad=True)], lr=1e-3)
    scheduler = get_adaptive_step_decay_schedule(
        optimizer, num_warmup_steps=10, decay_period=-1
    )

    # Warmup phase
    for _ in range(10):
        scheduler.step()

    # After warmup, LR should stay constant since decay_period=-1
    lr_after_warmup = scheduler.get_last_lr()[0]
    for _ in range(100):
        scheduler.step()

    lr_final = scheduler.get_last_lr()[0]
    assert lr_after_warmup == lr_final == 1e-3, f"LR should stay at 1e-3 when decay_period=-1, got {lr_final}"


def test_adaptive_step_decay_warmup():
    """Test warmup phase behavior."""
    from torch.optim import AdamW
    from verl.utils.torch_functional import get_adaptive_step_decay_schedule

    optimizer = AdamW([torch.randn(10, requires_grad=True)], lr=1e-3)
    num_warmup_steps = 100
    scheduler = get_adaptive_step_decay_schedule(
        optimizer, num_warmup_steps=num_warmup_steps, decay_period=50
    )

    # Check LR increases linearly during warmup
    prev_lr = 0.0
    for step in range(num_warmup_steps):
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        assert current_lr > prev_lr, f"LR should increase at step {step}"
        prev_lr = current_lr

    # After warmup, LR should be exactly initial LR
    scheduler.step()
    assert scheduler.get_last_lr()[0] == 1e-3, f"LR after warmup should be 1e-3"


def test_adaptive_step_decay_ratio_sequence():
    """Test the exact halving sequence."""
    from torch.optim import AdamW
    from verl.utils.torch_functional import get_adaptive_step_decay_schedule

    initial_lr = 1e-3
    optimizer = AdamW([torch.randn(10, requires_grad=True)], lr=initial_lr)
    decay_period = 20
    scheduler = get_adaptive_step_decay_schedule(
        optimizer, num_warmup_steps=0, decay_period=decay_period, min_lr_ratio=0.125  # 12.5%
    )

    expected_lrs = [
        1e-3,  # 0 halvings
        5e-4,  # 1 halving (step 20)
        2.5e-4,  # 2 halvings (step 40)
        1.25e-4,  # 3 halvings (step 60) - this is 12.5%, should stop here
    ]

    # Step through and verify each halving point
    step = 0
    scheduler.step()  # Step 1, LR = initial
    assert scheduler.get_last_lr()[0] == expected_lrs[0]

    # First decay_period
    for _ in range(decay_period):
        scheduler.step()
    step += decay_period + 1
    assert scheduler.get_last_lr()[0] == expected_lrs[1], f"LR after {step} steps should be {expected_lrs[1]}"

    # Second decay_period
    for _ in range(decay_period):
        scheduler.step()
    step += decay_period
    assert scheduler.get_last_lr()[0] == expected_lrs[2], f"LR after {step} steps should be {expected_lrs[2]}"

    # Third decay_period - should stop at min_lr_ratio
    for _ in range(decay_period):
        scheduler.step()
    step += decay_period
    assert scheduler.get_last_lr()[0] == expected_lrs[3], f"LR after {step} steps should be {expected_lrs[3]}"

    # Fourth decay_period - should stay at min_lr_ratio
    for _ in range(decay_period):
        scheduler.step()
    step += decay_period
    assert scheduler.get_last_lr()[0] == expected_lrs[3], f"LR should stay at min_lr_ratio {expected_lrs[3]}"
