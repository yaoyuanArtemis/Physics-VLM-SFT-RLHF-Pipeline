#!/bin/bash
################################################################################
# Qwen2.5-VL SFT LoRA 权重合并脚本
################################################################################

set -e

echo "================================================================================"
echo "⚙️  开始合并 SFT LoRA 权重到基础模型"
echo "================================================================================"

# 合并配置文件
CONFIG_FILE="SFT/configs/merge.yaml"

# 检查配置文件是否存在
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ 错误: 找不到配置文件 $CONFIG_FILE"
    echo "请确保在项目根目录运行此脚本"
    exit 1
fi

# 检查 LLaMA-Factory 是否安装
if ! command -v llamafactory-cli &> /dev/null; then
    echo "❌ 错误: 未找到 llamafactory-cli"
    echo "请先安装 LLaMA-Factory"
    exit 1
fi

# 显示配置信息
echo "📋 合并配置:"
echo "   - 基础模型: models/Qwen2.5-VL-7B-Instruct"
echo "   - LoRA 权重: output/qwen_physics_v1"
echo "   - 输出路径: models/qwen2_5_vl_physics_merged"
echo "   - 合并设备: CPU（节省显存）"
echo ""

# 启动合并
echo "🔧 开始合并权重..."
echo "⏳ 这可能需要几分钟，请耐心等待..."
echo "================================================================================"

llamafactory-cli export "$CONFIG_FILE"

echo "================================================================================"
echo "✅ LoRA 权重合并完成！"
echo "================================================================================"
echo ""
echo "📦 合并后的完整模型保存在: models/qwen2_5_vl_physics_merged/"
echo ""
echo "🔧 下一步操作:"
echo "  - 推理测试: bash SFT/inference_merged.sh"
echo "  - 或用于 RLHF 训练: bash RLHF-Training/run_swift_grpo.sh"
echo "================================================================================"
