# SFT 训练目录

这个目录包含 Supervised Fine-Tuning (SFT) 相关的配置和脚本。

## 文件说明

### 启动脚本（推荐使用）

- **train_sft.sh** - SFT 训练启动脚本
- **merge_sft.sh** - LoRA 权重合并脚本
- **inference_sft.sh** - 推理启动脚本（使用 LoRA 适配器）
- **inference_merged.sh** - 推理启动脚本（使用完整模型）

### 配置文件

- **train_qwen_physics.yaml** - SFT 训练配置文件
- **merge_sft_lora.yaml** - LoRA 权重合并配置（LLaMA-Factory export）
- **inference_sft_physics.yaml** - 推理配置（使用 LoRA 适配器，开发测试用）
- **inference_merged_model.yaml** - 推理配置（使用完整模型，生产部署用）
- **dataset_info.json** - 数据集配置文件（需要复制到 LLaMA-Factory/data/ 目录）

### Python 脚本

- **merge_sft_model.py** - 使用 PEFT 库合并 LoRA 权重的脚本（备用方案）

## 环境准备

### 1. 安装 LLaMA-Factory

```bash
# 克隆 LLaMA-Factory
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory

# 安装依赖
pip install -e .
```

### 2. 配置数据集

将本目录的 `dataset_info.json` 中的 `material_physics` 配置添加到 `LLaMA-Factory/data/dataset_info.json`：

```bash
# 方法1: 手动复制配置到 LLaMA-Factory/data/dataset_info.json

# 方法2: 使用软链接
ln -s $(pwd)/SFT/dataset_info.json LLaMA-Factory/data/material_physics_dataset.json
```

### 3. 下载基础模型

```bash
# 下载 Qwen2.5-VL-7B-Instruct 模型
# 可以从 Hugging Face 或 ModelScope 下载
# 将模型放置到 models/ 目录下
```

## 训练流程

### 1. SFT 训练

**方式A: 使用启动脚本（推荐）**

```bash
# 在项目根目录运行
bash SFT/train_sft.sh
```

**方式B: 使用命令行**

```bash
cd LLaMA-Factory
PYTORCH_ALLOC_CONF=expandable_segments:True llamafactory-cli train ../SFT/train_qwen_physics.yaml
```

训练配置说明：
- **模型**: Qwen2.5-VL-7B-Instruct
- **方法**: LoRA (rank=4, alpha=4)
- **数据集**: material_physics (25,000 张物理材料图片)
- **显存优化**: BF16 + Flash Attention 2 + Paged AdamW 8-bit
- **Batch Size**: 1 (gradient accumulation=16)
- **训练轮数**: 3 epochs

### 2. 合并 LoRA 权重

训练完成后，有三种方式合并权重：

#### 方法1: 使用启动脚本（推荐）

```bash
# 在项目根目录运行
bash SFT/merge_sft.sh
```

**优点**: 一键合并，自动检查环境

#### 方法2: 使用 LLaMA-Factory 命令

```bash
cd LLaMA-Factory
llamafactory-cli export ../SFT/merge_sft_lora.yaml
```

**优点**: 简单快捷，使用官方工具

#### 方法3: 使用 PEFT 库

```bash
python SFT/merge_sft_model.py
```

**注意**: 需要根据实际路径修改脚本中的：
- `BASE_MODEL_PATH` - 基础模型路径
- `LORA_ADAPTER_PATH` - LoRA 权重输出路径
- `OUTPUT_PATH` - 合并后模型保存路径

**优点**: 更灵活，可以自定义合并逻辑

### 3. 模型推理测试

#### 使用 LoRA 适配器推理（无需合并）

如果只是想快速测试，可以直接加载 LoRA 适配器。

**方式A: 使用启动脚本（推荐）**

```bash
# 在项目根目录运行
bash SFT/inference_sft.sh
```

**方式B: 使用命令行**

```bash
cd LLaMA-Factory
llamafactory-cli webchat ../SFT/inference_sft_physics.yaml
```

这会启动一个 WebUI 界面，可以上传物理材料图片进行测试。

#### 使用合并后的完整模型推理

**方式A: 使用启动脚本（推荐）**

```bash
# 在项目根目录运行
bash SFT/inference_merged.sh
```

**方式B: 使用 LLaMA-Factory WebUI**

```bash
cd LLaMA-Factory
llamafactory-cli webchat ../SFT/inference_merged_model.yaml
```

**方式C: 使用 Transformers 库（Python 代码）**

合并后的模型可以直接使用 Transformers 库加载：

```python
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image

# 加载模型
model = AutoModelForVision2Seq.from_pretrained(
    "/path/to/models/qwen2_5_vl_physics_merged",
    trust_remote_code=True,
    device_map="auto"
)

processor = AutoProcessor.from_pretrained(
    "/path/to/models/qwen2_5_vl_physics_merged",
    trust_remote_code=True
)

# 推理
image = Image.open("material_image.jpg")
prompt = "<image>请分析这个材料的微观结构特征"

inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=512)
response = processor.decode(output[0], skip_special_tokens=True)
print(response)
```

## 依赖版本

关键依赖版本（实测稳定）：
- `transformers==5.2.0`
- `torch>=2.0.0`
- `flash-attn` (编译安装)
- `bitsandbytes`

## 硬件要求

- GPU: RTX 4090 (24GB) 或更高
- CPU: 15+ 核心
- 内存: 64GB+ 推荐
- 存储: 100GB+ (模型 + 数据集)

## 注意事项

1. **LLaMA-Factory 不包含在本项目中**，需要单独安装
2. **模型文件不包含在本项目中**，需要单独下载
3. 训练前确保数据集路径正确（`data/material_train.json`）
4. 使用 `PYTORCH_ALLOC_CONF=expandable_segments:True` 避免显存碎片化
5. 训练时间约 12 小时（RTX 4090）

## 参考资源

- LLaMA-Factory: https://github.com/hiyouga/LLaMA-Factory
- Qwen2.5-VL: https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct
- 本项目 LoRA 权重: https://huggingface.co/yaoyuanlf/qwen2.5-vl-physics-lora
