#!/usr/bin/env python
"""
直接使用 PEFT 合并 SFT LoRA 权重到基础模型
避免 LLaMA-Factory 的 transformers.video_utils 依赖问题
"""

import os
import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForVision2Seq, AutoTokenizer, AutoProcessor

# 配置路径 - 确保在 . 目录结构下
BASE_MODEL_PATH = "./models/Qwen2.5-VL-7B-Instruct"
LORA_ADAPTER_PATH = "./output/qwen_physics_v1"
OUTPUT_PATH = "./models/qwen2_5_vl_physics_merged"  # 合并后模型放在 models 目录下

print("=" * 80)
print("🚀 开始合并 SFT LoRA 权重到基础模型")
print("=" * 80)

# 1. 加载基础模型配置
print(f"\n📖 步骤 1: 加载 LoRA 配置")
print(f"   Adapter 路径: {LORA_ADAPTER_PATH}")
peft_config = PeftConfig.from_pretrained(LORA_ADAPTER_PATH)
print(f"   ✅ LoRA 配置加载成功")
print(f"   - Base model: {peft_config.base_model_name_or_path}")
print(f"   - Task type: {peft_config.task_type}")
print(f"   - LoRA r: {peft_config.r}")
print(f"   - LoRA alpha: {peft_config.lora_alpha}")

# 2. 加载基础模型（使用 bfloat16 以节省内存并匹配 Flash Attention 要求）
print(f"\n🔧 步骤 2: 加载基础模型")
print(f"   模型路径: {BASE_MODEL_PATH}")
print(f"   数据类型: bfloat16")
print(f"   这可能需要几分钟...")

base_model = AutoModelForVision2Seq.from_pretrained(
    BASE_MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
    attn_implementation="flash_attention_2"  # 使用 Flash Attention 2.0
)
print(f"   ✅ 基础模型加载成功")

# 3. 加载 LoRA 权重
print(f"\n🎯 步骤 3: 加载 LoRA 适配器")
model = PeftModel.from_pretrained(
    base_model,
    LORA_ADAPTER_PATH,
    torch_dtype=torch.bfloat16,
)
print(f"   ✅ LoRA 适配器加载成功")

# 4. 合并权重
print(f"\n⚡ 步骤 4: 合并 LoRA 权重到基础模型")
print(f"   执行 merge_and_unload()...")
merged_model = model.merge_and_unload()
print(f"   ✅ 权重合并完成")

# 5. 保存合并后的模型
print(f"\n💾 步骤 5: 保存合并后的模型")
print(f"   输出路径: {OUTPUT_PATH}")
os.makedirs(OUTPUT_PATH, exist_ok=True)

# 保存模型权重
merged_model.save_pretrained(
    OUTPUT_PATH,
    safe_serialization=True,  # 使用 safetensors 格式
    max_shard_size="5GB"  # 分片大小，避免单个文件过大
)
print(f"   ✅ 模型权重保存完成")

# 6. 保存 tokenizer 和 processor
print(f"\n📝 步骤 6: 保存 tokenizer 和 processor")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
tokenizer.save_pretrained(OUTPUT_PATH)
print(f"   ✅ Tokenizer 保存完成")

processor = AutoProcessor.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
processor.save_pretrained(OUTPUT_PATH)
print(f"   ✅ Processor 保存完成")

# 7. 验证配置文件
print(f"\n🔍 步骤 7: 验证配置文件")
import json
config_path = os.path.join(OUTPUT_PATH, "config.json")
with open(config_path, "r") as f:
    config = json.load(f)

# 确保关键配置正确（保持原有的 architectures，不要修改）
# config["architectures"] 保持不变，让模型自动识别
if "model_type" not in config:
    config["model_type"] = "qwen2_5_vl"  # Qwen2.5-VL 的正确类型
config["torch_dtype"] = "bfloat16"

# 移除可能导致问题的字段
if "generation_config" in config:
    del config["generation_config"]
    print(f"   ⚠️  已移除 generation_config 字段（避免类型错误）")

# 保存修正后的配置
with open(config_path, "w") as f:
    json.dump(config, f, indent=2, ensure_ascii=False)
print(f"   ✅ 配置文件验证并修正完成")

# 8. 创建独立的 generation_config.json
print(f"\n📄 步骤 8: 创建 generation_config.json")
generation_config = {
    "do_sample": True,
    "temperature": 0.7,
    "top_p": 0.9,
    "max_new_tokens": 512,
    "eos_token_id": tokenizer.eos_token_id,
    "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
    "transformers_version": "4.46.0"
}
gen_config_path = os.path.join(OUTPUT_PATH, "generation_config.json")
with open(gen_config_path, "w") as f:
    json.dump(generation_config, f, indent=2)
print(f"   ✅ generation_config.json 创建完成")

print("\n" + "=" * 80)
print("🎉 模型合并完成！")
print("=" * 80)
print(f"\n合并后的模型保存在: {OUTPUT_PATH}")
print("\n包含文件:")
print("  ✅ config.json            - 模型配置")
print("  ✅ generation_config.json - 生成配置")
print("  ✅ model-*.safetensors    - 模型权重（分片）")
print("  ✅ tokenizer_config.json  - Tokenizer 配置")
print("  ✅ preprocessor_config.json - Processor 配置")
print("\n现在可以使用此模型进行 RL 训练了！")
print("=" * 80)
