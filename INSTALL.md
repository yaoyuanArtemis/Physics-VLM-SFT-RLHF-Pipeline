# 环境安装指南

本文档提供详细的环境配置步骤，确保能在 RTX 4090 (24GB) 或类似环境上成功运行训练。

## 硬件要求

### 最低配置
- **GPU**: RTX 4090 (24GB) 或 RTX 3090 (24GB)
- **CPU**: 15+ 核心
- **内存**: 64GB+
- **存储**: 100GB+ 可用空间

### 推荐配置
- **GPU**: RTX 4090 (24GB) x 2
- **CPU**: 32+ 核心
- **内存**: 128GB+
- **存储**: 500GB+ SSD

## 软件要求

- **操作系统**: Linux (Ubuntu 20.04/22.04 推荐)
- **Python**: 3.10 - 3.12
- **CUDA**: 12.1 - 12.4
- **Driver**: NVIDIA 驱动 >= 525

## 安装步骤

### 1. 创建虚拟环境（推荐）

```bash
# 使用 conda
conda create -n physics-vlm python=3.12
conda activate physics-vlm

# 或使用 venv
python3.12 -m venv venv
source venv/bin/activate
```

### 2. 安装 PyTorch（CUDA 版本）

```bash
# CUDA 12.1 版本
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# 验证安装
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')"
```

### 3. 安装 Flash Attention（关键步骤）

⚠️ **重要**: Flash Attention 需要编译安装，约需 15 分钟

```bash
# 方法1: 从源码编译（推荐）
pip install flash-attn==2.8.3 --no-build-isolation

# 方法2: 如果编译失败，尝试预编译版本
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu123torch2.5cxx11abiFALSE-cp312-cp312-linux_x86_64.whl

# 验证安装
python -c "import flash_attn; print(f'Flash Attention: {flash_attn.__version__}')"
```

### 4. 安装核心依赖

```bash
# 安装主要依赖
pip install -r requirements.txt

# 如果遇到依赖冲突，尝试:
pip install -r requirements.txt --no-deps
```

### 5. 安装 LLaMA-Factory（SFT 训练）

```bash
# 克隆仓库
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory

# 以开发模式安装
pip install -e .

# 验证安装
llamafactory-cli version
```

### 6. 验证环境

运行环境检查脚本：

```python
# check_env.py
import sys
import torch
import transformers
import peft
import flash_attn
from swift import Swift

print("=" * 80)
print("Environment Check")
print("=" * 80)
print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"CUDA Version: {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
print(f"Transformers: {transformers.__version__}")
print(f"PEFT: {peft.__version__}")
print(f"Flash Attention: {flash_attn.__version__}")
print(f"SWIFT: {Swift.__version__}")
print("=" * 80)
```

运行：
```bash
python check_env.py
```

## 常见问题

### Q1: Flash Attention 编译失败

**症状**: `error: command 'gcc' failed` 或 `CUDA kernel compilation error`

**解决方案**:
```bash
# 安装编译工具
sudo apt-get update
sudo apt-get install build-essential

# 确认 CUDA 路径
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH

# 重新安装
pip uninstall flash-attn -y
pip install flash-attn==2.8.3 --no-build-isolation
```

### Q2: Transformers 版本冲突

**症状**: `AttributeError: 'Qwen2VLModel' has no attribute 'visual'`

**解决方案**:
```bash
# 锁定到测试过的版本
pip install transformers==5.4.0 --force-reinstall
```

### Q3: OOM (Out of Memory) 显存不足

**症状**: `CUDA out of memory`

**解决方案**:
1. 检查 YAML 配置:
   - `per_device_train_batch_size: 1`
   - `gradient_accumulation_steps: 16`
   - `bf16: true`
   - `optim: paged_adamw_8bit`

2. 启用显存碎片整理:
   ```bash
   export PYTORCH_ALLOC_CONF=expandable_segments:True
   ```

### Q4: vLLM 安装失败

**症状**: `Failed building wheel for vllm`

**解决方案**:
```bash
# vLLM 对 CUDA 版本敏感，确认匹配
pip install vllm==0.7.2 --extra-index-url https://download.pytorch.org/whl/cu121
```

### Q5: 网络问题（国内用户）

**症状**: `Connection timeout` 或下载缓慢

**解决方案**:
```bash
# 使用镜像源
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/path/to/hf_cache

# PyPI 镜像
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 环境变量配置

创建 `.env` 文件或添加到 `~/.bashrc`:

```bash
# CUDA 配置
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Hugging Face 配置
export HF_ENDPOINT=https://hf-mirror.com  # 国内镜像
export HF_HOME=/path/to/hf_cache
export HF_TOKEN=your_hf_token  # 可选

# PyTorch 优化
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TORCH_COMPILE_DISABLE=1  # 禁用编译缓存（稳定性）

# 训练优化
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=true
```

## 磁盘空间规划

确保有足够的存储空间：

```
/path/to/project/
├── models/                    # ~30GB
│   ├── Qwen2.5-VL-7B-Instruct    (~15GB)
│   └── qwen2_5_vl_physics_merged (~16GB)
├── data/                      # ~10GB
│   ├── material_dataset/         (~8GB)
│   └── physics_rl_train.parquet  (~200KB)
├── outputs/                   # ~50GB (训练输出)
├── .cache/                    # ~20GB (Hugging Face 缓存)
└── logs/                      # ~1GB
```

**总计**: 建议预留 **150GB+** 空间

## 下一步

环境配置完成后：

1. 📖 阅读 [SFT/README.md](SFT/README.md) 进行 SFT 训练
2. 🎯 阅读 [RLHF-Training/README.md](RLHF-Training/README.md) 进行 RLHF 训练
3. 📊 查看 [paper_results/](paper_results/) 了解实验结果

## 技术支持

如遇到问题：
1. 检查 [常见问题](#常见问题) 章节
2. 提交 [GitHub Issue](https://github.com/yourusername/your-repo/issues)
3. 查看项目主 README 文档
