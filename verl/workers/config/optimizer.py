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
import warnings
from dataclasses import dataclass
from typing import Optional

from omegaconf import MISSING

from verl.base_config import BaseConfig

__all__ = ["OptimizerConfig", "FSDPOptimizerConfig", "McoreOptimizerConfig", "build_optimizer", "VeOmniOptimizerConfig", "AdaptiveStepDecayOptimizerConfig"]


@dataclass
class OptimizerConfig(BaseConfig):
    """Base optimizer configuration.

    Args:
        lr (float): learning rate. Must be specified.
        lr_warmup_steps_ratio (float): Warmup steps ratio; total steps will be injected at runtime.
        total_training_steps (int): Total training steps (must be overridden at runtime).
        weight_decay (float): Weight decay factor.
        lr_warmup_steps (Optional[int]): Number of warmup steps; None delegates to lr_warmup_steps_ratio.
    """

    _mutable_fields = {"clip_grad", "total_training_steps", "lr_warmup_steps"}

    lr: float = 1e-3
    lr_warmup_steps_ratio: float = 0.0
    total_training_steps: int = -1
    weight_decay: float = 0.01
    lr_warmup_steps: Optional[int] = -1
    betas: tuple[float, float] = (0.9, 0.999)
    clip_grad: float = 1.0
    # deprecate grad_clip
    grad_clip: Optional[float] = None

    def __post_init__(self):
        assert self.lr != MISSING
        if self.grad_clip is not None:
            warnings.warn("`grad_clip` is deprecated, use `clip_grad` instead.", DeprecationWarning, stacklevel=2)
            self.clip_grad = self.grad_clip


@dataclass
class VeOmniOptimizerConfig(OptimizerConfig):
    """VeOmni optimizer configuration extending base OptimizerConfig.

    Args:
        optimizer (str): Optimizer name; default is "adamw".
        lr (float): Learning rate.
        lr_min (float): Minimum learning rate.
        lr_start (float): Starting learning rate for warmup.
        lr_decay_ratio (float): LR decay ratio.
        lr_scheduler_type (str): LR scheduler type: "constant" or "cosine".
    """

    _mutable_fields = OptimizerConfig._mutable_fields.copy()

    optimizer: str = "adamw"
    lr_min: float = 0.0
    lr_start: float = 0.0
    lr_decay_ratio: float = 1.0
    lr_scheduler_type: str = "constant"
    override_optimizer_config: Optional[dict] = None


@dataclass
class FSDPOptimizerConfig(OptimizerConfig):
    """FSDP optimizer configuration extending base OptimizerConfig.

    Args:
        optimizer (str): Optimizer class name (e.g., "AdamW", "AdamW8bit", "_AdamW").
        optimizer_impl (str): Module path to import optimizer from (e.g., "torch.optim", "torchao.optim",
            "bitsandbytes.optim").
        lr (float): Learning rate.
        min_lr_ratio (Optional[float]): Minimum LR ratio for cosine schedule.
        lr_scheduler_type (str): LR scheduler type: "constant" or "cosine".
        num_cycles (float): Number of cosine cycles in LR schedule.
    """

    _mutable_fields = OptimizerConfig._mutable_fields.copy()
    _mutable_fields.add("lr_scheduler_type")

    optimizer: str = "AdamW"
    optimizer_impl: str = "torch.optim"
    min_lr_ratio: Optional[float] = None
    # deprecate warmup_style
    warmup_style: Optional[str] = None
    lr_scheduler_type: str = "constant"
    num_cycles: float = 0.5
    override_optimizer_config: Optional[dict] = None

    def __post_init__(self):
        if self.warmup_style is not None:
            assert self.warmup_style in ["constant", "cosine"]
            warnings.warn(
                "`warmup_style` is deprecated, use `lr_scheduler_type` instead.", DeprecationWarning, stacklevel=2
            )
            self.lr_scheduler_type = self.warmup_style
        # Support "adaptive_step_decay" as valid type
        valid_types = ["constant", "cosine", "adaptive_step_decay"]
        if self.lr_scheduler_type not in valid_types:
            raise ValueError(f"lr_scheduler_type must be one of {valid_types}, got {self.lr_scheduler_type}")
        return super().__post_init__()


@dataclass
class AdaptiveStepDecayOptimizerConfig(FSDPOptimizerConfig):
    """Adaptive Step-Decay optimizer configuration.

    Extends FSDPOptimizerConfig with parameters for response length
    surge detection and dynamic LR decay. This scheduler is designed
    to stabilize RL training by detecting response length surge and
    triggering LR decay to prevent training collapse.

    Algorithm (from paper arXiv:2602.01826v1):
    1. Monitor response_length/mean during training
    2. Detect surge when response_length > surge_threshold * baseline
    3. Set decay_period = 1.8 * surge_step
    4. Halve LR every decay_period steps until reaching min_lr_ratio * initial LR

    Args:
        lr_scheduler_type (str): Must be "adaptive_step_decay".
        decay_period (int): Steps between LR halving. -1 means auto-compute from surge.
            After surge detection: decay_period = 1.8 * surge_step.
        min_lr_ratio (float): Minimum LR ratio (default 0.1, i.e., 10% of initial LR).
        surge_threshold_multiplier (float): Response length surge threshold multiplier.
            Surge is detected when response_length > baseline * surge_threshold_multiplier.
        surge_baseline_window (int): Number of steps to compute baseline response length.
        surge_cooldown_steps (int): Minimum steps between surge detections to avoid noise.
        surge_detected (bool): Whether surge has been detected. Set to True after detection.
        surge_step (int): The step at which surge was detected. -1 means not yet detected.
    """

    _mutable_fields = FSDPOptimizerConfig._mutable_fields.copy()
    _mutable_fields.update({"decay_period", "surge_detected", "surge_step"})

    lr_scheduler_type: str = "adaptive_step_decay"
    decay_period: int = -1  # Auto-computed: 1.8 * surge_step
    min_lr_ratio: float = 0.1
    surge_threshold_multiplier: float = 3.0
    surge_baseline_window: int = 50  # Steps to compute baseline
    surge_cooldown_steps: int = 50
    surge_detected: bool = False
    surge_step: int = -1

    def __post_init__(self):
        if self.lr_scheduler_type != "adaptive_step_decay":
            warnings.warn(
                f"AdaptiveStepDecayOptimizerConfig created with lr_scheduler_type={self.lr_scheduler_type}, "
                f"overriding to 'adaptive_step_decay'",
                UserWarning,
                stacklevel=2,
            )
            self.lr_scheduler_type = "adaptive_step_decay"
        assert self.min_lr_ratio > 0 and self.min_lr_ratio < 1.0, (
            f"min_lr_ratio must be between 0 and 1 (exclusive), got {self.min_lr_ratio}"
        )
        assert self.surge_threshold_multiplier > 1.0, (
            f"surge_threshold_multiplier must be > 1.0, got {self.surge_threshold_multiplier}"
        )
        assert self.surge_baseline_window > 0, (
            f"surge_baseline_window must be > 0, got {self.surge_baseline_window}"
        )
        return super().__post_init__()


@dataclass
class McoreOptimizerConfig(OptimizerConfig):
    """Mcore optimizer configuration extending base OptimizerConfig.

    Args:
        optimizer (str): Optimizer name; default is "adam".
        lr (float): Learning rate.
        clip_grad (float): Gradient clipping norm.
        lr_warmup_init (float): Initial learning rate for warmup; defaults to 0.0.
        lr_decay_steps (Optional[int]): Number of decay steps.
        lr_decay_style (str): LR decay style: "constant", "linear", "cosine", or "inverse_square_root".
        min_lr (float): Minimum learning rate.
        weight_decay_incr_style (str): Weight decay increment style: "constant" or "cosine".
        lr_wsd_decay_style (str): Weight-standard-deviation decay style: "constant", "exponential", or "cosine".
        lr_wsd_decay_steps (Optional[int]): Number of steps for weight-standard-deviation decay.
        use_checkpoint_opt_param_scheduler (bool): Whether to use checkpoint optimizer parameter scheduler.
    """

    optimizer: str = "adam"
    lr_warmup_init: float = 0.0
    lr_decay_steps: Optional[int] = None
    lr_decay_style: str = "linear"
    min_lr: float = 0.0
    weight_decay_incr_style: str = "constant"
    lr_wsd_decay_style: str = "exponential"
    lr_wsd_decay_steps: Optional[int] = None
    use_checkpoint_opt_param_scheduler: bool = False
    override_optimizer_config: Optional[dict] = None


def build_optimizer(parameters, config: FSDPOptimizerConfig):
    """Build an optimizer based on the configuration.

    Dynamically imports and instantiates an optimizer class from the specified module.

    Args:
        parameters: Model parameters to optimize
        config: FSDPOptimizerConfig with optimizer settings

    Returns:
        Optimizer instance

    Examples:
        # PyTorch AdamW
        config.optimizer_impl = "torch.optim"
        config.optimizer = "AdamW"

        # TorchAO AdamW with bf16 stochastic rounding
        config.optimizer_impl = "torchao.optim"
        config.optimizer = "_AdamW"
        config.override_optimizer_config = {"bf16_stochastic_round": True}

        # BitsAndBytes AdamW 8bit
        config.optimizer_impl = "bitsandbytes.optim"
        config.optimizer = "AdamW8bit"
    """
    import importlib

    optimizer_args = {
        "lr": config.lr,
        "weight_decay": config.weight_decay,
    }

    optimizer_name_lower = config.optimizer.lower()
    if "adam" in optimizer_name_lower or "ademamix" in optimizer_name_lower:
        optimizer_args["betas"] = config.betas

    if config.override_optimizer_config is not None:
        optimizer_args.update(config.override_optimizer_config)

    try:
        module = importlib.import_module(config.optimizer_impl)
        optimizer_cls = getattr(module, config.optimizer)
    except ImportError as e:
        raise ImportError(
            f"Failed to import module '{config.optimizer_impl}'. Make sure the package is installed. Error: {e}"
        ) from e
    except AttributeError as e:
        raise AttributeError(
            f"Optimizer '{config.optimizer}' not found in module '{config.optimizer_impl}'. "
            f"Available optimizers: {dir(module)}"
        ) from e

    return optimizer_cls(parameters, **optimizer_args)
