#!/bin/bash
# ==============================================================================
# Adaptive Step-Decay Algorithm Reproduction Script
# Paper: "Beyond Precision: Training-Inference Mismatch is an Optimization Problem"
# arXiv: 2602.01826v1
#
# Model: Qwen3-4B-Base on DAPO dataset with NPU (torch_npu)
# Expected: surge_step ~110, decay_period ~198, AIME 2024 acc 50%+
# ==============================================================================

set -xeuo pipefail

# ------------------------------------------------------------------------------
# NPU Environment Setup
# ------------------------------------------------------------------------------
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

# vLLM configuration for NPU
export VLLM_USE_V1=1
export VLLM_VERSION=0.9.1

# NPU performance optimizations
export TASK_QUEUE_ENABLE=2
export CPU_AFFINITY_CONF=1
export LD_PRELOAD="/usr/lib/aarch64-linux-gnu/libjemalloc.so.2${LD_PRELOAD:+:$LD_PRELOAD}"

# ------------------------------------------------------------------------------
# Training Configuration
# ------------------------------------------------------------------------------
project_name='Adaptive_DAPO'
exp_name='Qwen3-4B-Adaptive-NPU'

# Hardware config
trainer_n_gpus_per_node=8
trainer_nnodes=1

# ------------------------------------------------------------------------------
# Paths Configuration
# ------------------------------------------------------------------------------
RAY_DATA_HOME=${RAY_DATA_HOME:-"${HOME}/verl"}
MODEL_PATH=${MODEL_PATH:-"${RAY_DATA_HOME}/models/Qwen3-4B-Base"}
CKPTS_DIR=${CKPTS_DIR:-"${RAY_DATA_HOME}/ckpts/${project_name}/${exp_name}"}
TRAIN_FILE=${TRAIN_FILE:-"${RAY_DATA_HOME}/data/dapo-math-17k.parquet"}
TEST_FILE=${TEST_FILE:-"${RAY_DATA_HOME}/data/aime-2024.parquet"}

export TENSORBOARD_DIR="${RAY_DATA_HOME}/tensorboard_dir/${project_name}/${exp_name}"
mkdir -p "${RAY_DATA_HOME}/logs/${project_name}"
mkdir -p "${TENSORBOARD_DIR}"
LOG_PATH="${RAY_DATA_HOME}/logs/${project_name}/${exp_name}.log"

# ------------------------------------------------------------------------------
# Algorithm Parameters (from Paper Appendix)
# ------------------------------------------------------------------------------
# Core hyperparameters from paper:
# - Initial LR: 1e-6
# - batch_size: 64
# - ppo_mini_batch_size: 64
# - ppo_epoch: 1
# - min_lr_ratio: 0.1
# - rollouts per prompt: 16
# - max_response_length: 8192

# DAPO specific parameters
adv_estimator=grpo
use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0
clip_ratio_low=0.2
clip_ratio_high=0.28

# Length configuration
max_prompt_length=2048
max_response_length=8192
enable_overlong_buffer=True
overlong_buffer_len=4096
overlong_penalty_factor=1.0

# Loss aggregation (token-level for DAPO)
loss_agg_mode="token-mean"

# Sampling configuration
n_resp_per_prompt=16  # rollouts per prompt from paper
train_prompt_bsz=64   # batch_size from paper

# Temperature for sampling
temperature=1.0
top_p=1.0
top_k=-1

# ------------------------------------------------------------------------------
# Adaptive Step-Decay Configuration
# ------------------------------------------------------------------------------
# These parameters control the surge detection and LR decay:
# - surge_threshold_multiplier: 3.0 (surge = baseline * 3)
# - surge_baseline_window: 50 (first 50 steps to compute baseline)
# - min_lr_ratio: 0.1 (final LR = initial * 0.1)

# Expected behavior:
# Step 0-10: Warmup (LR: 0 -> 1e-6)
# Step 11-110: Constant (waiting for surge)
# Step ~110: Surge detected, decay_period = 198
# Step 111+: LR halving every 198 steps until 1e-7

# ------------------------------------------------------------------------------
# Performance Configuration
# ------------------------------------------------------------------------------
sp_size=2
use_dynamic_bsz=True
actor_ppo_max_token_len=$(( (max_prompt_length + max_response_length) / sp_size ))
infer_ppo_max_token_len=$(( (max_prompt_length + max_response_length) / sp_size ))
offload=True
gen_tp=2

# ------------------------------------------------------------------------------
# Run Training
# ------------------------------------------------------------------------------
echo "========================================"
echo "Adaptive Step-Decay Training on NPU"
echo "Model: Qwen3-4B-Base"
echo "Dataset: DAPO-math-17k"
echo "Expected surge_step: ~110"
echo "Expected decay_period: ~198"
echo "========================================"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.lr_scheduler_type=adaptive_step_decay \
    algorithm.min_lr_ratio=0.1 \
    algorithm.surge_threshold_multiplier=3.0 \
    algorithm.surge_baseline_window=50 \
    algorithm.surge_cooldown_steps=50 \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    algorithm.filter_groups.enable=False \
    data.train_files=${TRAIN_FILE} \
    data.val_files=${TEST_FILE} \
    data.prompt_key=prompt \
    data.truncation='left' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.train_batch_size=${train_prompt_bsz} \
    data.gen_batch_size=$((train_prompt_bsz * 3)) \
    data.filter_overlong_prompts=True \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_scheduler_type=adaptive_step_decay \
    actor_rollout_ref.actor.optim.min_lr_ratio=0.1 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_bsz} \
    actor_rollout_ref.actor.ppo_epoch=1 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    actor_rollout_ref.actor.fsdp_config.forward_prefetch=True \
    actor_rollout_ref.actor.entropy_checkpointing=True \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.90 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.ref.use_torch_compile=False \
    actor_rollout_ref.ref.entropy_checkpointing=True \
    actor_rollout_ref.ref.fsdp_config.forward_prefetch=True \
    +actor_rollout_ref.model.override_config.attention_dropout=0. \
    +actor_rollout_ref.model.override_config.embd_pdrop=0. \
    +actor_rollout_ref.model.override_config.resid_pdrop=0. \
    reward_model.reward_manager=dapo \
    reward_model.overlong_buffer.enable=${enable_overlong_buffer} \
    reward_model.overlong_buffer.len=${overlong_buffer_len} \
    reward_model.overlong_buffer.penalty_factor=${overlong_penalty_factor} \
    trainer.critic_warmup=0 \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${exp_name} \
    trainer.logger=['console','tensorboard'] \
    trainer.default_local_dir=${CKPTS_DIR} \
    trainer.n_gpus_per_node=${trainer_n_gpus_per_node} \
    trainer.nnodes=${trainer_nnodes} \
    trainer.val_before_train=False \
    trainer.test_freq=10 \
    trainer.save_freq=20 \
    trainer.total_epochs=1 \
    trainer.total_training_steps=100 \
    trainer.resume_mode=auto \
    data.shuffle=False 2>&1 | tee ${LOG_PATH}

echo "========================================"
echo "Training completed. Check logs at:"
echo "  ${LOG_PATH}"
echo "Check checkpoints at:"
echo "  ${CKPTS_DIR}"
echo "Check tensorboard at:"
echo "  ${TENSORBOARD_DIR}"
echo "========================================"