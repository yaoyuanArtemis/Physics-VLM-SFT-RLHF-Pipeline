#!/bin/bash
################################################################################
# Qwen2.5-VL 物理学 SFT 训练脚本
################################################################################

set -e

echo "================================================================================"
echo "🚀 开始 Qwen2.5-VL 物理学 SFT 训练"
echo "================================================================================"

# 训练配置文件
CONFIG_FILE="SFT/configs/train.yaml"

# 检查配置文件是否存在
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ 错误: 找不到配置文件 $CONFIG_FILE"
    echo "请确保在项目根目录运行此脚本"
    exit 1
fi

# 检查 LLaMA-Factory 是否安装
if ! command -v llamafactory-cli &> /dev/null; then
    echo "❌ 错误: 未找到 llamafactory-cli"
    echo "请先安装 LLaMA-Factory:"
    echo "  git clone https://github.com/hiyouga/LLaMA-Factory.git"
    echo "  cd LLaMA-Factory && pip install -e ."
    exit 1
fi

# 环境变量设置
echo "📋 设置训练环境变量..."
export PYTORCH_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=true

# 显示配置信息
echo "✅ 环境变量已设置"
echo "   - PYTORCH_ALLOC_CONF=expandable_segments:True"
echo "   - OMP_NUM_THREADS=8"
echo "   - 配置文件: $CONFIG_FILE"
echo ""

# 启动训练
echo "🔥 启动 SFT 训练..."
echo "================================================================================"

llamafactory-cli train "$CONFIG_FILE"

echo "================================================================================"
echo "✅ SFT 训练完成！"
echo "================================================================================"
echo ""
echo "📦 LoRA 权重保存在: output/qwen_physics_v1/"
echo ""
echo "🔧 下一步操作:"
echo "  1. 合并权重: bash SFT/merge_sft.sh"
echo "  2. 推理测试: bash SFT/inference_sft.sh"
echo "================================================================================"
