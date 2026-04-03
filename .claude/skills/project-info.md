---
name: project-info
description: Display Physics VLM project information including GitHub repository and Hugging Face model links
---

# Physics VLM SFT & RLHF Pipeline

## 📦 GitHub Repository
**Main Project**: https://github.com/yaoyuanArtemis/Physics-VLM-SFT-RLHF-Pipeline

Complete training pipeline for fine-tuning Qwen2.5-VL on physics material microstructure analysis.

---

## 🤗 Hugging Face Models

### RLHF Model (Final Version)
**Qwen2.5-VL-7B-Physics-RLHF**: https://huggingface.co/yaoyuanlf/Qwen2.5-VL-7B-Physics-RLHF

Full model trained with GRPO (Group Relative Policy Optimization) algorithm using custom physics reward function. Ready for inference.

### SFT Model (LoRA Adapter)
**Qwen2.5-VL-Physics-LoRA**: https://huggingface.co/yaoyuanlf/qwen2.5-vl-physics-lora

LoRA adapter from supervised fine-tuning phase. Requires base model to use.

---

## 📊 Dataset

**Physics VLM Dataset**: https://huggingface.co/datasets/yaoyuanlf/physics-vlm-dataset

25,000+ physics material microstructure images with expert annotations.

---

## 🚀 Quick Start

```bash
# Clone repository
git clone https://github.com/yaoyuanArtemis/Physics-VLM-SFT-RLHF-Pipeline.git
cd Physics-VLM-SFT-RLHF-Pipeline

# Install dependencies
pip install -r requirements.txt

# Use RLHF model directly
from transformers import AutoModelForVision2Seq, AutoProcessor

model = AutoModelForVision2Seq.from_pretrained(
    "yaoyuanlf/Qwen2.5-VL-7B-Physics-RLHF",
    trust_remote_code=True
)
processor = AutoProcessor.from_pretrained(
    "yaoyuanlf/Qwen2.5-VL-7B-Physics-RLHF",
    trust_remote_code=True
)
```

---

## 📖 Documentation

- **Installation Guide**: `INSTALL.md`
- **SFT Training**: `SFT/README.md`
- **RLHF Training**: `RLHF-Training/README.md`

---

## 🎯 Key Features

- ✅ **SFT Phase**: LoRA fine-tuning on 25K material images
- ✅ **RLHF Phase**: GRPO with custom physics reward function
- ✅ **Single GPU**: Optimized for RTX 4090 (24GB)
- ✅ **Production Ready**: Full model deployment ready

---

## 📄 Citation

If you use this project in your research, please cite:

```bibtex
@misc{physics-vlm-2026,
  author = {Yaoyuan Liu},
  title = {Physics VLM: Fine-tuning Vision-Language Models for Material Microstructure Analysis},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/yaoyuanArtemis/Physics-VLM-SFT-RLHF-Pipeline}
}
```

---

**Project Status**: ✅ Complete | **Last Updated**: April 2026
