# Adaptive Step-Decay 论文Baseline复现指南

## 概述

本文档提供在华为昇腾NPU上使用torch_npu复现论文"Beyond Precision: Training-Inference Mismatch is an Optimization Problem" (arXiv:2602.01826v1)中Adaptive Step-Decay算法的完整步骤。

**目标**: 在Qwen3-4B-Base模型上验证算法有效性，复现论文报告的训练稳定性改善和AIME 2024准确率。

---

## 1. 环境准备

### 1.1 硬件要求

| 设备 | 状态 | 说明 |
|------|------|------|
| Atlas 200T A2 Box16 | ✅ | 8卡NPU服务器 |
| Atlas 900 A2 PODc | ✅ | 大规模集群 |
| Atlas 800T A3 | ✅ | 高性能训练服务器 |

### 1.2 软件版本

```bash
# 必需软件版本
Python >= 3.10, < 3.12
CANN == 8.3.RC1
torch == 2.7.1
torch_npu == 2.7.1
torchvision == 0.22.1
triton-ascend == 3.2.0rc4
vllm == v0.11.0
vllm-ascend == v0.11.0rc1
```

### 1.3 Docker环境 (推荐)

```bash
# 拉取预构建镜像
docker pull swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:8.3.rc1-a3-ubuntu22.04-py3.11

# 运行容器
docker run -it --device=/dev/davinci0 --device=/dev/davinci1 \
    --device=/dev/davinci2 --device=/dev/davinci3 \
    --device=/dev/davinci4 --device=/dev/davinci5 \
    --device=/dev/davinci6 --device=/dev/davinci7 \
    --device=/dev/davinci_manager \
    -v /usr/local/Ascend:/usr/local/Ascend \
    -v ${HOME}/verl:/root/verl \
    --name verl-npu \
    swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:8.3.rc1-a3-ubuntu22.04-py3.11

# 或自行构建
cd verl/docker/ascend
docker build -f Dockerfile.ascend_8.3.rc1_a3 -t verl-npu:latest .
```

### 1.4 手动安装

```bash
# 1. 激活CANN环境
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

# 2. 安装torch_npu
pip install torch==2.7.1 torch_npu==2.7.1 torchvision==0.22.1

# 3. 安装triton-ascend
pip uninstall -y triton triton-ascend
pip install triton-ascend==3.2.0rc4

# 4. 安装vllm-ascend
git clone --depth 1 --branch v0.11.0 https://github.com/vllm-project/vllm.git
cd vllm && VLLM_TARGET_DEVICE=empty pip install -v -e . && cd ..

git clone --depth 1 --branch v0.11.0rc1 https://github.com/vllm-project/vllm-ascend.git
cd vllm-ascend && pip install -v -e . && cd ..

# 5. 安装MindSpeed (可选，用于Megatron后端)
git clone https://gitcode.com/Ascend/MindSpeed.git
cd MindSpeed && git checkout f2b0977e && cd ..
git clone --depth 1 --branch core_v0.12.1 https://github.com/NVIDIA/Megatron-LM.git
pip install -e MindSpeed
pip install mbridge
export PYTHONPATH=$PYTHONPATH:"$(pwd)/Megatron-LM"

# 6. 安装verl
git clone https://github.com/volcengine/verl.git
cd verl && pip install -r requirements-npu.txt && pip install -v -e .
```

---

## 2. 数据准备

### 2.1 DAPO数据集

DAPO (Decoupled Clip and Dynamic Sampling Policy Optimization) 数据集包含约13k数学问题样本。

```bash
# 使用verl提供的脚本自动下载
cd verl/recipe/dapo
bash prepare_dapo_data.sh

# 数据将下载到:
# ${HOME}/verl/data/dapo-math-17k.parquet (训练集)
# ${HOME}/verl/data/aime-2024.parquet (测试集)
```

### 2.2 数据格式验证

```bash
# 检查数据文件
python3 -c "
import pandas as pd
train_df = pd.read_parquet('${HOME}/verl/data/dapo-math-17k.parquet')
test_df = pd.read_parquet('${HOME}/verl/data/aime-2024.parquet')
print(f'Train samples: {len(train_df)}')
print(f'Test samples: {len(test_df)}')
print(f'Train columns: {train_df.columns.tolist()}')
"
```

---

## 3. 模型准备

### 3.1 下载Qwen3-4B-Base

```bash
# 方式一: 使用huggingface-cli
pip install huggingface_hub[hf_transfer]
export HF_HUB_ENABLE_HF_TRANSFER=1
huggingface-cli download Qwen/Qwen3-4B --local-dir ${HOME}/verl/models/Qwen3-4B-Base

# 方式二: 使用git
git clone https://huggingface.co/Qwen/Qwen3-4B ${HOME}/verl/models/Qwen3-4B-Base

# 验证模型文件
ls -la ${HOME}/verl/models/Qwen3-4B-Base/
```

---

## 4. 训练脚本配置

### 4.1 使用提供的脚本

脚本位置: `verl/recipe/dapo/run_dapo_qwen3_4b_adaptive_npu.sh`

```bash
# 设置环境变量
export RAY_DATA_HOME=${HOME}/verl
export MODEL_PATH=${RAY_DATA_HOME}/models/Qwen3-4B-Base
export TRAIN_FILE=${RAY_DATA_HOME}/data/dapo-math-17k.parquet
export TEST_FILE=${RAY_DATA_HOME}/data/aime-2024.parquet

# 运行训练
cd verl/recipe/dapo
bash run_dapo_qwen3_4b_adaptive_npu.sh
```

### 4.2 论文超参数对照

脚本已配置论文Appendix Table中的所有超参数:

| 参数 | 论文值 | 脚本配置 |
|------|--------|----------|
| `lr` | 1e-6 | `actor_rollout_ref.actor.optim.lr=1e-6` ✅ |
| `batch_size` | 64 | `data.train_batch_size=64` ✅ |
| `ppo_mini_batch_size` | 64 | `actor_rollout_ref.actor.ppo_mini_batch_size=64` ✅ |
| `ppo_epoch` | 1 | `actor_rollout_ref.actor.ppo_epoch=1` ✅ |
| `min_lr_ratio` | 0.1 | `algorithm.min_lr_ratio=0.1` ✅ |
| `rollouts_per_prompt` | 16 | `actor_rollout_ref.rollout.n=16` ✅ |
| `max_response_length` | 8192 | `data.max_response_length=8192` ✅ |

### 4.3 Adaptive Step-Decay配置

```bash
# 核心参数
algorithm.lr_scheduler_type=adaptive_step_decay  # 启用算法
algorithm.min_lr_ratio=0.1                       # 最小LR=初始*0.1
algorithm.surge_threshold_multiplier=3.0         # 激增阈值=baseline*3
algorithm.surge_baseline_window=50               # 前50步计算baseline
algorithm.surge_cooldown_steps=50                # 检测冷却期

# Actor端配置 (需同步)
actor_rollout_ref.actor.optim.lr_scheduler_type=adaptive_step_decay
actor_rollout_ref.actor.optim.min_lr_ratio=0.1
```

---

## 5. 预期训练行为

### 5.1 Surge检测时机

根据论文数据:
- Qwen3-4B: surge_step ≈ **110** (训练步数)
- 计算公式: decay_period = 1.8 × surge_step
- 预期: decay_period ≈ **198** (约200步)

### 5.2 LR衰减序列

| 训练阶段 | 步数范围 | LR值 | 说明 |
|----------|----------|------|------|
| Warmup | 0-10 | 0 → 1e-6 | 线性增长 |
| Waiting | 11-110 | 1e-6 | 等待surge |
| Surge | ~110 | 检测触发 | decay_period=198 |
| Period 0 | 111-308 | 1e-6 | 无衰减 |
| Period 1 | 309-506 | 5e-7 | 首次减半 |
| Period 2 | 507-704 | 2.5e-7 | 二次减半 |
| Period 3 | 705+ | 1e-7 | 达到min_lr |

### 5.3 训练稳定性指标

训练过程中应观察:
- `response_length/mean`: 触发前稳定，~110步出现激增
- `log_ppl_abs_diff`: 保持低位 (<0.1为良好)
- `aime_2024/acc`: 最终应达50%+

---

## 6. 监控与验证

### 6.1 TensorBoard监控

```bash
# 启动TensorBoard
tensorboard --logdir ${HOME}/verl/tensorboard_dir/Adaptive_DAPO --port 6006

# 在浏览器访问: http://localhost:6006
```

### 6.2 关键监控指标

| Metric | 说明 | 预期值 |
|--------|------|--------|
| `response_length/mean` | 平均响应长度 | step ~110激增 |
| `adaptive_step_decay/surge_detected` | 激增触发状态 | step ~110变为True |
| `adaptive_step_decay/decay_period` | 衰减周期值 | ~198-204 |
| `adaptive_step_decay/baseline` | 响应长度基线 | 前50步平均值 |
| `actor/lr` | 当前学习率 | 按预期序列衰减 |
| `aime_2024/acc` | AIME准确率 | 最终50%+ |

### 6.3 日志分析

```bash
# 查看完整日志
tail -f ${HOME}/verl/logs/Adaptive_DAPO/Qwen3-4B-Adaptive-NPU.log

# 搜索关键事件
grep "Surge detected" ${HOME}/verl/logs/Adaptive_DAPO/*.log
grep "decay_period" ${HOME}/verl/logs/Adaptive_DAPO/*.log
grep "adaptive_step_decay" ${HOME}/verl/logs/Adaptive_DAPO/*.log

# 查看LR变化
grep "actor/lr" ${HOME}/verl/logs/Adaptive_DAPO/*.log
```

### 6.4 Checkpoint状态验证

```bash
# 检查adaptive state文件
cat ${HOME}/verl/ckpts/Adaptive_DAPO/Qwen3-4B-Adaptive-NPU/global_step_*/adaptive_step_decay_state.json

# 内容示例:
# {
#   "surge_detected": true,
#   "surge_step": 112,
#   "decay_period": 201,
#   "surge_baseline": 2048.5,
#   "response_length_history": [...]
# }
```

---

## 7. 验证成功标准

训练成功的判定标准:

1. **Surge检测**: step ~110成功触发surge检测
2. **Decay计算**: decay_period正确计算 (约198-204)
3. **LR衰减**: LR按预期序列衰减 (1e-6 → 5e-7 → 2.5e-7 → 1e-7)
4. **Checkpoint**: adaptive_step_decay_state.json正确保存/加载
5. **准确率**: AIME 2024准确率达到50%或以上
6. **稳定性**: 训练过程稳定，无奖励崩溃

---

## 8. 故障排查

### 8.1 常见问题

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| Surge未触发 | threshold太高 | 调低surge_threshold_multiplier至2.5 |
| LR不衰减 | decay_period=-1 | 确认surge检测成功，检查RPC调用 |
| NPU OOM | 内存不足 | 减小batch_size或增大offload |
| Checkpoint恢复失败 | state文件损坏 | 检查JSON格式，手动修复 |
| 训练崩溃 | 数据问题 | 检查数据集格式和reward计算 |

### 8.2 调试命令

```bash
# 检查NPU状态
npu-smi info

# 检查vllm-ascend状态
python3 -c "import vllm_ascend; print(vllm_ascend.__version__)"

# 检查torch_npu
python3 -c "import torch_npu; print(torch_npu.__version__)"

# 测试adaptive scheduler
python3 -c "
from verl.utils.torch_functional import get_adaptive_step_decay_schedule
import torch
optimizer = torch.optim.Adam([torch.randn(10)])
scheduler = get_adaptive_step_decay_schedule(optimizer, 10, 198, 0.1)
for i in range(500):
    scheduler.step()
    if i % 50 == 0:
        print(f'Step {i}: LR={optimizer.param_groups[0][\"lr\"]}')
"
```

---

## 9. 进阶配置

### 9.1 Megatron后端 (可选)

如需使用Megatron后端替代FSDP:

```bash
# 添加以下参数
actor_rollout_ref.actor.strategy=megatron
+actor_rollout_ref.actor.megatron.override_transformer_config.use_flash_attn=True
```

### 9.2 多节点训练

```bash
# 配置多节点
trainer_nnodes=4
trainer_n_gpus_per_node=8

# 启动Ray集群
ray start --head --port=6379
ray start --address='HEAD_NODE_IP:6379'
```

### 9.3 自定义参数调整

```bash
# 调整激增阈值 (更敏感)
algorithm.surge_threshold_multiplier=2.5

# 调整基线窗口 (更快响应)
algorithm.surge_baseline_window=30

# 调整冷却期 (减少误检)
algorithm.surge_cooldown_steps=100
```

---

## 10. 相关资源

- **论文**: arXiv:2602.01826v1
- **DAPO主页**: https://dapo-sia.github.io/
- **verl仓库**: https://github.com/volcengine/verl
- **torch_npu文档**: https://gitcode.com/Ascend/pytorch
- **vllm-ascend**: https://github.com/vllm-project/vllm-ascend
- **MindSpeed**: https://gitcode.com/Ascend/MindSpeed

---

## 附录: 完整命令流程

```bash
# 1. 进入Docker容器或激活环境
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

# 2. 准备数据
cd ${HOME}/verl/recipe/dapo && bash prepare_dapo_data.sh

# 3. 下载模型
huggingface-cli download Qwen/Qwen3-4B \
    --local-dir ${HOME}/verl/models/Qwen3-4B-Base

# 4. 设置环境变量
export RAY_DATA_HOME=${HOME}/verl
export MODEL_PATH=${RAY_DATA_HOME}/models/Qwen3-4B-Base
export TRAIN_FILE=${RAY_DATA_HOME}/data/dapo-math-17k.parquet
export TEST_FILE=${RAY_DATA_HOME}/data/aime-2024.parquet

# 5. 运行训练
cd ${HOME}/verl/recipe/dapo
bash run_dapo_qwen3_4b_adaptive_npu.sh

# 6. 监控训练
tensorboard --logdir ${HOME}/verl/tensorboard_dir/Adaptive_DAPO --port 6006

# 7. 查看结果
grep "aime_2024/acc" ${HOME}/verl/logs/Adaptive_DAPO/*.log
```