#!/bin/bash
################################################################################
# RLHF 模型推理脚本 - 启动 WebUI 进行交互式测试
################################################################################

set -e

echo "================================================================================"
echo "🎯 启动 RLHF 模型推理 WebUI"
echo "================================================================================"

# 检查 SWIFT 是否安装
if ! command -v swift &> /dev/null; then
    echo "❌ 错误: 未找到 swift 命令"
    echo "请先安装 SWIFT: pip install ms-swift[all]"
    exit 1
fi

# 选择推理模式
echo "请选择推理模式:"
echo "  1. 使用合并后的完整模型（推荐，速度快）"
echo "  2. 使用基础模型 + LoRA 适配器（灵活）"
echo ""
read -p "请输入选项 (1/2): " MODE

if [ "$MODE" = "1" ]; then
    # 完整模型推理
    MODEL_PATH="models/Qwen2_5_VL_Physics_RLHF_Merged"

    if [ ! -d "$MODEL_PATH" ]; then
        echo "❌ 错误: 找不到合并后的模型 $MODEL_PATH"
        echo "请先运行: bash RLHF-Training/export_rlhf.sh"
        exit 1
    fi

    echo ""
    echo "📋 推理配置:"
    echo "   - 模型: $MODEL_PATH (完整模型)"
    echo ""
    echo "🌐 启动 WebUI..."
    echo "================================================================================"

    swift infer \
        --model_type qwen2_5_vl \
        --model_id_or_path "$MODEL_PATH" \
        --template qwen2_5_vl

elif [ "$MODE" = "2" ]; then
    # LoRA 适配器推理
    BASE_MODEL="models/Qwen2.5-VL-7B-Instruct"
    ADAPTER_PATH="outputs/swift_grpo/checkpoint-1500"

    if [ ! -d "$ADAPTER_PATH" ]; then
        echo "❌ 错误: 找不到适配器 $ADAPTER_PATH"
        echo "请检查训练输出目录"
        exit 1
    fi

    echo ""
    echo "📋 推理配置:"
    echo "   - 基础模型: $BASE_MODEL"
    echo "   - 适配器: $ADAPTER_PATH"
    echo ""
    echo "🌐 启动 WebUI..."
    echo "================================================================================"

    swift infer \
        --model_type qwen2_5_vl \
        --model_id_or_path "$BASE_MODEL" \
        --adapters "$ADAPTER_PATH" \
        --template qwen2_5_vl
else
    echo "❌ 无效的选项"
    exit 1
fi

echo ""
echo "================================================================================"
echo "💡 使用说明:"
echo "   1. 在浏览器中打开显示的 URL"
echo "   2. 上传物理材料图片"
echo "   3. 输入问题进行测试"
echo "   4. 按 Ctrl+C 停止服务"
echo "================================================================================"
