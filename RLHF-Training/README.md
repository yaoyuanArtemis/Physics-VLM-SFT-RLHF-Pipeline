# Reinforcement Learning (RLHF) 配置目录

这个目录包含基于 SWIFT 框架的 GRPO (Group Relative Policy Optimization) 强化学习训练配置。

## 文件说明

### 启动脚本

- **run_swift_grpo.sh** - GRPO 训练启动脚本（主要使用）
  - 单卡训练
  - 禁用 vLLM 加速
  - 完整的训练参数配置

- **train_physics_swift.sh** - 训练启动脚本（备用）
  - 多卡训练支持
  - 启用 vLLM 加速

- **export_rlhf.sh** - 模型导出脚本
  - 合并 LoRA 权重到基础模型
  - 生成完整的独立模型

- **inference_rlhf.sh** - 推理测试脚本
  - 启动 WebUI 进行交互式测试
  - 支持完整模型或 LoRA 适配器

### 自定义奖励函数

- **swift_physics_reward.py** - 物理材料学专用打分器
  - 基于关键词匹配的领域奖励机制
  - 正向奖励：核心视觉特征、计算属性关联
  - 负向惩罚：尺度混乱、晶体学错误、灌水词汇
  - 自动注入到 SWIFT 框架的 `orms` 字典

### 工具脚本

- **convert_data.py** - 数据格式转换工具
- **start_ui.py** - WebUI 启动脚本（备用）

## 环境准备

### 安装 SWIFT 框架

```bash
# 安装 ms-swift
pip install ms-swift[all]

# 或从源码安装
git clone https://github.com/modelscope/swift.git
cd swift
pip install -e .
```

### 依赖版本
- Python >= 3.8
- PyTorch >= 2.0.0
- transformers >= 4.37.0
- ms-swift >= 2.0.0

## 数据准备

RLHF 训练数据位于 `data/` 目录：
- **data/physics_rl_train.parquet** - Parquet 格式（推荐，179KB）
- **data/physics_rl_train.jsonl** - JSONL 格式（7.0MB）

数据格式示例：
```json
{
  "prompt": "<image>请详细描述这个材料的微观组织特征",
  "solution": "该材料呈现典型的多晶结构，可观察到明显的晶界...",
  "image": "data/material_dataset/mat_img_0.jpg"
}
```

## 训练流程

### 1. 确保 SFT 模型已准备

RLHF 训练需要基于 SFT 微调后的模型：

```bash
# 确保已完成 SFT 训练并合并权重
ls models/qwen2_5_vl_physics_merged/
```

### 2. 启动 GRPO 训练

**方式A: 使用启动脚本（推荐）**

```bash
# 在项目根目录运行
bash RLHF-Training/run_swift_grpo.sh
```

**方式B: 修改脚本参数后运行**

如果需要自定义参数，可以编辑 `run_swift_grpo.sh`：

```bash
--model models/qwen2_5_vl_physics_merged \
--dataset data/physics_rl_train.parquet \
--output_dir outputs/swift_grpo \
```

### 3. 导出完整模型

训练完成后，将 LoRA 权重合并到基础模型：

**方式A: 使用导出脚本（推荐）**

```bash
# 在项目根目录运行
bash RLHF-Training/export_rlhf.sh
```

**注意**: 需要先修改脚本中的 `ADAPTER_PATH`，指向训练输出的最佳 checkpoint：
```bash
ADAPTER_PATH="outputs/swift_grpo/v42-20260401-162133/checkpoint-1500"
```

**方式B: 使用命令行**

```bash
swift export \
    --model_type qwen2_5_vl \
    --model_id_or_path models/Qwen2.5-VL-7B-Instruct \
    --adapters outputs/swift_grpo/checkpoint-1500 \
    --merge_lora true \
    --output_dir models/Qwen2_5_VL_Physics_RLHF_Merged
```

### 4. 模型推理测试

**方式A: 使用推理脚本（推荐）**

```bash
# 在项目根目录运行
bash RLHF-Training/inference_rlhf.sh
```

脚本会提供两个选项：
1. 使用合并后的完整模型（速度快）
2. 使用基础模型 + LoRA 适配器（灵活）

**方式B: 手动启动 WebUI**

```bash
# 使用完整模型
swift infer \
    --model_type qwen2_5_vl \
    --model_id_or_path models/Qwen2_5_VL_Physics_RLHF_Merged \
    --template qwen2_5_vl

# 或使用 LoRA 适配器
swift infer \
    --model_type qwen2_5_vl \
    --model_id_or_path models/Qwen2.5-VL-7B-Instruct \
    --adapters outputs/swift_grpo/checkpoint-1500 \
    --template qwen2_5_vl
```

### 5. 训练配置说明

关键参数：
```bash
--rlhf_type grpo                      # 使用 GRPO 算法
--model_type qwen2_5_vl               # Qwen2.5-VL 模型
--reward_funcs material_physics_score # 自定义奖励函数
--num_sample_generations 4            # 每个 prompt 生成 4 个候选
--learning_rate 1e-6                  # 学习率
--per_device_train_batch_size 1       # 批次大小
--gradient_accumulation_steps 16      # 梯度累积
--max_length 1024                     # 最大长度
--num_train_epochs 1                  # 训练轮数
--use_vllm false                      # 是否使用 vLLM 加速
--bf16 true                           # 使用 BF16
--temperature 0.8                     # 生成温度
```

## 奖励函数机制

`swift_physics_reward.py` 实现了专门的物理材料学打分系统：

### 正向奖励

1. **核心视觉特征识别** (+0.5/关键词)
   - 晶界 (grain boundary)
   - 位错 (dislocation)
   - 孪晶 (twin)
   - 析出相 (precipitate)
   - 层片结构 (lamellar)
   - 等...

2. **计算方法关联** (+0.8/关键词)
   - DFT / 第一性原理
   - 分子动力学 (MD)
   - 相场模拟 (phase field)
   - 深度势能 (Deep Potential)
   - 等...

3. **精确匹配奖励** (+3.0)
   - 生成文本完全包含标准答案

4. **长度奖励** (+0~0.5)
   - 鼓励详细的专业描述

### 负向惩罚

1. **尺度混乱** (-2.0)
   ```python
   # 例：将原子级特征描述为宏观尺度
   if "atomistic" in ground_truth and "macroscopic" in completion:
       score -= 2.0
   ```

2. **晶体学错误** (-2.5)
   ```python
   # 例：BCC/HCP 相变混淆
   if "bcc" in GT and "hcp" in prediction:
       score -= 2.5
   ```

3. **灌水词汇** (-1.5)
   ```python
   # 禁止使用非技术性描述
   bs_keywords = ["artistic", "beautiful", "aesthetic"]
   ```

### 最终得分范围

- 最低分：-5.0（严重错误）
- 最高分：根据关键词匹配累积（通常 0~10 分）

## 训练监控

SWIFT 会自动输出：
- 每步的 Reward 分数
- Loss 曲线
- 生成样本质量

建议配合 TensorBoard 查看：
```bash
tensorboard --logdir=/path/to/rl_outputs/swift_grpo
```

## 硬件要求

- **GPU**: RTX 4090 (24GB) 或更高
- **CPU**: 16+ 核心
- **内存**: 64GB+ 推荐
- **存储**: 50GB+（模型 + checkpoint）

### 显存优化建议

如果显存不足：
1. 减小 `--per_device_train_batch_size`
2. 增加 `--gradient_accumulation_steps`
3. 减小 `--max_length`
4. 设置 `--use_vllm false`（禁用 vLLM）
5. 减少 `--num_sample_generations`（候选数）

## 训练输出

训练完成后，输出目录包含：
```
rl_outputs/swift_grpo/
├── checkpoint-100/          # 训练检查点
├── checkpoint-200/
├── ...
├── final_checkpoint/        # 最终模型
├── training_args.json       # 训练参数
└── trainer_state.json       # 训练状态
```

## 模型推理

使用 RLHF 训练后的模型：

```bash
# 启动 WebUI
swift infer \
    --model_type qwen2_5_vl \
    --model /path/to/rl_outputs/swift_grpo/final_checkpoint \
    --template qwen2_5_vl
```

## 注意事项

1. ✅ **SWIFT 框架需要单独安装**，不包含在本项目中
2. ✅ **必须先完成 SFT 训练**，RLHF 是在 SFT 模型基础上进行
3. ✅ **数据格式必须正确**，包含 `prompt`、`solution`、`image` 字段
4. ✅ **奖励函数可自定义**，根据实际任务调整打分规则
5. ⚠️ **GRPO 算法无需 Reference 模型**，比 PPO 更节省显存
6. ⚠️ **训练时间较长**，建议使用云平台的自动关机功能

## 参考资源

- SWIFT 框架: https://github.com/modelscope/swift
- GRPO 论文: https://arxiv.org/abs/2402.03300
- Qwen2.5-VL: https://github.com/QwenLM/Qwen2.5-VL
- 本项目模型: https://huggingface.co/yaoyuanlf/qwen2.5-vl-physics-rlhf
