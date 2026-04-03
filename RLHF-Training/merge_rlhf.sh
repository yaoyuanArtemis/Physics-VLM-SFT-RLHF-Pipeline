#!/bin/bash
################################################################################
# RLHF 模型导出脚本 - 合并 GRPO 训练的 LoRA 权重到基础模型
################################################################################

set -e

echo "================================================================================"
echo "⚙️  导出 RLHF 训练后的完整模型"
echo "================================================================================"

# 检查 SWIFT 是否安装
if ! command -v swift &> /dev/null; then
    echo "❌ 错误: 未找到 swift 命令"
    echo "请先安装 SWIFT:"
    echo "  pip install ms-swift[all]"
    exit 1
fi

# 配置路径（请根据实际情况修改）
BASE_MODEL="models/Qwen2.5-VL-7B-Instruct"
ADAPTER_PATH="outputs/swift_grpo/checkpoint-1500"  # 训练输出的最佳 checkpoint
OUTPUT_DIR="models/Qwen2_5_VL_Physics_RLHF_Merged"

# 检查适配器路径是否存在
if [ ! -d "$ADAPTER_PATH" ]; then
    echo "❌ 错误: 找不到适配器路径 $ADAPTER_PATH"
    echo ""
    echo "请检查训练输出目录，找到最佳 checkpoint，例如:"
    echo "  outputs/swift_grpo/v42-20260401-162133/checkpoint-1500"
    echo ""
    echo "修改此脚本中的 ADAPTER_PATH 变量"
    exit 1
fi

# 显示配置信息
echo "📋 导出配置:"
echo "   - 基础模型: $BASE_MODEL"
echo "   - 适配器路径: $ADAPTER_PATH"
echo "   - 输出目录: $OUTPUT_DIR"
echo ""

# 执行导出
echo "🔧 开始导出模型..."
echo "⏳ 这可能需要几分钟，请耐心等待..."
echo "================================================================================"

swift export \
    --model_type qwen2_5_vl \
    --model_id_or_path "$BASE_MODEL" \
    --adapters "$ADAPTER_PATH" \
    --merge_lora true \
    --output_dir "$OUTPUT_DIR"

echo "================================================================================"
echo "✅ RLHF 模型导出完成！"
echo "================================================================================"
echo ""
echo "📦 完整模型保存在: $OUTPUT_DIR"
echo ""
echo "🔧 下一步操作:"
echo "  1. 推理测试: bash RLHF-Training/inference_rlhf.sh"
echo "  2. 上传到 HuggingFace"
echo "================================================================================"
