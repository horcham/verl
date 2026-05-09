# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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

import math
import torch
from megatron.core.optimizer import OptimizerConfig
from megatron.core.optimizer import get_megatron_optimizer as get_megatron_optimizer_native
from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler

from verl.utils.logger import print_rank_0


class AdaptiveStepDecayScheduler:
    """
    Wrapper scheduler implementing Adaptive Step-Decay for Megatron backend.

    Megatron's OptimizerParamScheduler doesn't support step-based halving directly,
    so this wrapper manages the halving logic internally and uses Megatron's
    constant mode for the base scheduler.

    Algorithm:
    1. Warmup: LR increases from init_lr to max_lr (handled by Megatron scheduler)
    2. Before surge detection (decay_period=-1): LR stays constant at max_lr
    3. After surge detection: LR halves every decay_period steps
    4. Minimum floor: LR stops at min_lr_ratio * max_lr

    The wrapper tracks the current step and computes the appropriate LR at each step(),
    then directly sets the optimizer's learning rate.

    Args:
        optimizer: Megatron optimizer instance
        config: Optimizer configuration (McoreOptimizerConfig)
        decay_period: Steps between LR halving (-1 = not yet set, awaiting surge)
        min_lr_ratio: Minimum LR ratio w.r.t max_lr (default 0.1)
    """

    def __init__(
        self,
        optimizer,
        config,
        decay_period: int = -1,
        min_lr_ratio: float = 0.1,
    ):
        self.optimizer = optimizer
        self.config = config
        self.decay_period = decay_period
        self.min_lr_ratio = min_lr_ratio
        self.current_step = 0
        self.max_lr = config.lr
        self.min_lr = config.lr * min_lr_ratio
        self._megatron_scheduler = None
        self._build_megatron_scheduler()

    def _build_megatron_scheduler(self):
        """Build the underlying Megatron scheduler with constant decay style."""
        lr_decay_steps = self.config.get("total_training_steps", 1000)
        lr_warmup_steps = self.config.get("lr_warmup_steps", 0)
        if lr_warmup_steps <= 0:
            lr_warmup_steps_ratio = self.config.get("lr_warmup_steps_ratio", 0.0)
            lr_warmup_steps = int(lr_warmup_steps_ratio * lr_decay_steps)

        # Use constant decay style - we'll manage LR manually after warmup
        self._megatron_scheduler = OptimizerParamScheduler(
            self.optimizer,
            init_lr=self.config.get("lr_warmup_init", 0.0),
            max_lr=self.max_lr,
            min_lr=self.min_lr,
            lr_warmup_steps=lr_warmup_steps,
            lr_decay_steps=lr_decay_steps,
            lr_decay_style="constant",  # We manage decay manually
            start_wd=self.config.get("weight_decay", 0.01),
            end_wd=self.config.get("weight_decay", 0.01),
            wd_incr_steps=lr_decay_steps,
            wd_incr_style="constant",
            use_checkpoint_opt_param_scheduler=False,
            override_opt_param_scheduler=True,
        )

        print_rank_0(
            f"AdaptiveStepDecayScheduler created: max_lr={self.max_lr}, "
            f"min_lr={self.min_lr}, decay_period={self.decay_period}"
        )

    def step(self, increment: int = 1):
        """
        Advance scheduler by increment steps and update LR if needed.

        This method:
        1. Advances the Megatron scheduler (handles warmup)
        2. After warmup, applies halving logic if decay_period is set
        """
        # Advance the Megatron scheduler for warmup handling
        self._megatron_scheduler.step(increment)
        self.current_step += increment

        # After warmup, apply our custom halving logic if decay_period is set
        lr_warmup_steps = self._megatron_scheduler.lr_warmup_steps
        if self.current_step >= lr_warmup_steps and self.decay_period > 0:
            steps_since_warmup = self.current_step - lr_warmup_steps
            num_halvings = steps_since_warmup // self.decay_period

            # Compute decayed LR: halve every decay_period steps
            decayed_lr = self.max_lr / (2 ** num_halvings)
            target_lr = max(self.min_lr, decayed_lr)

            # Directly set optimizer LR (override Megatron's constant scheduler)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = target_lr

    def get_last_lr(self):
        """Get current learning rate from optimizer."""
        return [param_group["lr"] for param_group in self.optimizer.param_groups]

    def get_lr(self):
        """Alias for get_last_lr() for compatibility."""
        return self.get_last_lr()

    def update_decay_period(self, decay_period: int):
        """
        Update decay_period (called when surge is detected).

        This doesn't rebuild the scheduler, just updates the decay_period
        parameter for subsequent step() calls.
        """
        self.decay_period = decay_period
        print_rank_0(f"AdaptiveStepDecayScheduler: Updated decay_period to {decay_period}")

    def state_dict(self):
        """Return scheduler state for checkpointing."""
        return {
            "current_step": self.current_step,
            "decay_period": self.decay_period,
            "min_lr_ratio": self.min_lr_ratio,
            "max_lr": self.max_lr,
            "min_lr": self.min_lr,
        }

    def load_state_dict(self, state_dict):
        """Load scheduler state from checkpoint."""
        self.current_step = state_dict["current_step"]
        self.decay_period = state_dict["decay_period"]
        self.min_lr_ratio = state_dict["min_lr_ratio"]
        self.max_lr = state_dict["max_lr"]
        self.min_lr = state_dict["min_lr"]


def init_megatron_optim_config(
    optim_config: dict, use_distributed_optimizer: bool = True, fp16: bool = False
) -> OptimizerConfig:
    optim_args = {
        "optimizer": optim_config.optimizer,
        "lr": optim_config.lr,
        "min_lr": optim_config.min_lr,
        "clip_grad": optim_config.clip_grad,
        "weight_decay": optim_config.weight_decay,
        "use_distributed_optimizer": use_distributed_optimizer,
    }
    if fp16:
        optim_args.update(
            {
                "bf16": False,
                "fp16": True,
                "params_dtype": torch.float16,
                "initial_loss_scale": 32768,
                "min_loss_scale": 1,
                "use_precision_aware_optimizer": True,
                "store_param_remainders": False,
            }
        )
    else:  # bf16 mode
        optim_args.update(
            {
                "bf16": True,
                "params_dtype": torch.bfloat16,
            }
        )
    override_config = optim_config.get("override_optimizer_config", {})
    if override_config:
        for k, v in override_config.items():
            optim_args[k] = v

    print_rank_0(f"optimizer config after override: {optim_args}")

    config = OptimizerConfig(**optim_args)
    return config


def get_megatron_optimizer(
    model,
    config: OptimizerConfig,
):
    # Base optimizer.
    return get_megatron_optimizer_native(
        config=config,
        model_chunks=model,
    )


def get_megatron_optimizer_param_scheduler(
    optimizer,
    config,
):
    """
    Get the optimizer parameter scheduler for Megatron.
    """
    lr_decay_steps = config.lr_decay_steps
    lr_warmup_steps = config.lr_warmup_steps
    if config.get("lr_decay_steps", None) is None:
        lr_decay_steps = config.total_training_steps
    wsd_decay_steps = None
    if config.get("lr_wsd_decay_steps", None) is not None:
        wsd_decay_steps = config.lr_wsd_decay_steps
    if config.get("lr_warmup_steps_ratio", None) is not None and (
        config.get("lr_warmup_steps", None) is None or config.lr_warmup_steps <= 0
    ):
        lr_warmup_steps = int(config.lr_warmup_steps_ratio * lr_decay_steps)

    opt_param_scheduler = OptimizerParamScheduler(
        optimizer,
        init_lr=config.lr_warmup_init,
        max_lr=config.lr,
        min_lr=config.min_lr,
        lr_warmup_steps=lr_warmup_steps,
        lr_decay_steps=lr_decay_steps,
        lr_decay_style=config.lr_decay_style,
        start_wd=config.weight_decay,
        end_wd=config.weight_decay,
        wd_incr_steps=config.total_training_steps,
        wd_incr_style=config.weight_decay_incr_style,
        use_checkpoint_opt_param_scheduler=config.use_checkpoint_opt_param_scheduler,
        override_opt_param_scheduler=(not config.use_checkpoint_opt_param_scheduler),
        wsd_decay_steps=wsd_decay_steps,
        lr_wsd_decay_style=config.lr_wsd_decay_style,
    )

    return opt_param_scheduler


def get_megatron_last_lr(optimizer):
    """
    Get the last learning rate from the optimizer parameter scheduler.
    """
    return optimizer.param_groups[0]["lr"]
