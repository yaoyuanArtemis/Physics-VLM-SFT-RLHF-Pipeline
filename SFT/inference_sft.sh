#!/bin/bash
################################################################################
# Qwen2.5-VL SFT 模型推理脚本（使用 LoRA 适配器）
################################################################################

set -e

echo "================================================================================"
echo "🎯 启动 SFT 模型推理（LoRA 适配器模式）"
echo "================================================================================"

# 推理配置文件
CONFIG_FILE="SFT/configs/inference_lora.yaml"

# 检查配置文件是否存在
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ 错误: 找不到配置文件 $CONFIG_FILE"
    echo "请确保在项目根目录运行此脚本"
    exit 1
fi

# 检查 LLaMA-Factory 是否安装
if ! command -v llamafactory-cli &> /dev/null; then
    echo "❌ 错误: 未找到 llamafactory-cli"
    exit 1
fi

# 显示配置信息
echo "📋 推理配置:"
echo "   - 基础模型: models/Qwen2.5-VL-7B-Instruct"
echo "   - LoRA 适配器: output/qwen_physics_v1"
echo "   - 推理模式: LoRA 适配器（开发测试用）"
echo ""

# 启动 WebUI
echo "🌐 启动 Web 推理界面..."
echo "================================================================================"
echo ""
echo "📌 使用说明:"
echo "   1. 等待模型加载完成"
echo "   2. 在浏览器中打开显示的 URL"
echo "   3. 上传物理材料图片"
echo "   4. 输入问题进行测试"
echo ""
echo "💡 提示: 按 Ctrl+C 停止服务"
echo "================================================================================"

llamafactory-cli webchat "$CONFIG_FILE"
