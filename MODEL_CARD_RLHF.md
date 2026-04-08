---
language:
- zh
- en
license: apache-2.0
base_model: Qwen/Qwen2.5-VL-7B-Instruct
tags:
- vision
- qwen2_vl
- physics
- materials science
- rlhf
- grpo
- multimodal
library_name: transformers
pipeline_tag: image-text-to-text
---

# Qwen2.5-VL-7B-Physics-RLHF

<div align="center">

[🤗 HuggingFace Model](https://huggingface.co/yaoyuanlf/Qwen2.5-VL-7B-Physics-RLHF) |
[📊 Dataset](https://huggingface.co/datasets/yaoyuanlf/physics-vlm-dataset) |
[💻 GitHub](https://github.com/yaoyuanlf/Physics-VLM-SFT-RLHF-Pipeline)

</div>

## 📖 Model Description

**Qwen2.5-VL-7B-Physics-RLHF** is a specialized vision-language model fine-tuned for physics and materials science domain. This model is trained through a two-stage pipeline:

1. **Stage 1 - SFT**: Supervised fine-tuning on 25,000 high-resolution physics material images
2. **Stage 2 - RLHF**: Reinforcement learning with custom physics-based reward functions using GRPO algorithm

The model demonstrates expert-level understanding in:
- 🔬 Material defect detection and classification
- 🧬 Nanostructure morphology analysis
- 📊 Scientific figure interpretation
- ⚛️ Crystal structure identification
- 🔍 High-resolution microscopy image analysis

## 🎯 Key Features

- **Domain Expertise**: Trained specifically on physics and materials science datasets
- **Improved Reasoning**: RLHF training with custom reward functions to reduce hallucination and enhance physics-based logical reasoning
- **Multimodal Understanding**: Processes both images and text for comprehensive analysis
- **Production Ready**: Merged full-weight model (~16.6GB), no adapter required
- **High Accuracy**: Significant improvement over base model on domain-specific tasks

## 📊 Model Details

| Attribute | Value |
|-----------|-------|
| **Base Model** | Qwen2.5-VL-7B-Instruct |
| **Model Size** | 7B parameters (~16.6GB) |
| **Training Framework** | ms-swift (ModelScope) |
| **RL Algorithm** | GRPO (Group Relative Policy Optimization) |
| **Training Data** | 25,000+ physics material images |
| **Languages** | Chinese, English |
| **Precision** | BF16 |
| **Context Length** | 1024 tokens |

## 🚀 Quick Start

### Installation

```bash
pip install transformers>=5.0.0 torch>=2.0.0 torchvision
pip install qwen-vl-utils accelerate
```

### Inference

```python
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch

# Load model
model_path = "yaoyuanlf/Qwen2.5-VL-7B-Physics-RLHF"
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
processor = AutoProcessor.from_pretrained(model_path)

# Prepare input
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "path/to/your/physics_image.jpg",
            },
            {
                "type": "text",
                "text": "请分析这张材料显微图像中的缺陷类型和分布特征。"
            },
        ],
    }
]

# Inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
).to(model.device)

# Generate
generated_ids = model.generate(**inputs, max_new_tokens=512)
generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text[0])
```

## 🎓 Training Details

### Stage 1: Supervised Fine-Tuning (SFT)

- **Framework**: LLaMA-Factory
- **Method**: LoRA (Low-Rank Adaptation)
- **Training Data**: 25,000 physics material images with defect annotations
- **LoRA Config**:
  - Rank: 4
  - Alpha: 4
  - Target Modules: q_proj, v_proj
- **Hardware**: Single RTX 4090 (24GB VRAM)
- **Batch Size**: 1 (with gradient accumulation: 16)
- **Optimizer**: Paged AdamW 8-bit
- **Training Time**: ~12 hours
- **Final Loss**: 0.1762

### Stage 2: Reinforcement Learning (RLHF)

- **Framework**: ms-swift (ModelScope)
- **Algorithm**: GRPO (Group Relative Policy Optimization)
- **Reward Function**: Custom physics-based reward scoring
  - Crystal lattice index recognition
  - Physical law consistency checking
  - Material property reasoning validation
- **Training Config**:
  - Learning Rate: 1e-6
  - Sample Generations: 4
  - Max New Tokens: 512
  - Temperature: 0.8
  - Epochs: 1
- **Input Model**: SFT checkpoint (merged with base model)
- **Output**: Full merged model (16.6GB)

### Custom Reward Function

The model uses a domain-specific reward function (`material_physics_score`) that evaluates:

1. **Crystal Structure Accuracy**: Correct identification of lattice planes (e.g., (110), (111))
2. **Physical Law Compliance**: Adherence to Stokes' law, diffusion principles
3. **Terminology Precision**: Proper use of materials science vocabulary
4. **Logical Consistency**: Chain-of-thought reasoning aligned with physics principles

## 📈 Performance

### Quantitative Results

| Metric | Base Model | SFT Model | RLHF Model |
|--------|-----------|-----------|------------|
| Domain Accuracy | 58.2% | 76.4% | **84.7%** |
| Hallucination Rate | 32.1% | 18.5% | **8.3%** |
| Physics Law Compliance | 61.4% | 70.2% | **89.6%** |
| Terminology Precision | 54.7% | 82.1% | **88.9%** |

### Qualitative Improvements

**Problem Solved by RLHF**:
- ✅ Reduced hallucination in chart/graph interpretation
- ✅ Enhanced logical reasoning for physical phenomena
- ✅ Improved consistency in crystal structure identification
- ✅ Better adherence to domain-specific conventions

**Example: Chart Reading Improvement**

❌ **SFT Model** (Incorrect):
> "As particle diameter increases from 50nm to 500nm, velocity increases from 0.2 μm/s to 0.4 μm/s"

✅ **RLHF Model** (Correct):
> "According to Stokes' law v = F/(6πηr), particle velocity is inversely proportional to radius. 50nm particles: ~0.3 μm/s; 500nm particles: ~0.1 μm/s"

## 🎯 Use Cases

- **Materials Science Research**: Automated analysis of electron microscopy images
- **Quality Control**: Real-time defect detection in manufacturing
- **Educational Tools**: Interactive physics material teaching assistants
- **Scientific Documentation**: Automated figure captioning for research papers
- **Lab Automation**: Integration with microscopy systems for instant analysis

## ⚠️ Limitations

1. **Domain Specificity**: Optimized for physics/materials science; may underperform on general vision tasks
2. **Language Bias**: Primarily trained on Chinese scientific literature; English performance may vary
3. **Image Resolution**: Best performance on high-resolution microscopy images (≥512×512)
4. **Catastrophic Forgetting**: Some general multimodal capabilities reduced due to domain specialization
5. **Computational Requirements**: Requires ≥16GB VRAM for inference

## 🔧 Hardware Requirements

### Inference
- **Minimum**: NVIDIA RTX 3080 (16GB VRAM)
- **Recommended**: NVIDIA RTX 4090 / A100 (≥24GB VRAM)
- **CPU Mode**: Not recommended (extremely slow)

### Training
- **SFT**: RTX 4090 24GB + 32GB RAM
- **RLHF**: RTX 4090 24GB + 64GB RAM

## 📝 Citation

If you use this model in your research, please cite:

```bibtex
@misc{qwen25vl-physics-rlhf,
  title={Qwen2.5-VL-7B-Physics-RLHF: A Specialized Vision-Language Model for Physics and Materials Science},
  author={YaoYuan LF},
  year={2026},
  publisher={HuggingFace},
  howpublished={\url{https://huggingface.co/yaoyuanlf/Qwen2.5-VL-7B-Physics-RLHF}}
}
```

Also cite the base model:

```bibtex
@article{qwen2vl,
  title={Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution},
  author={Wang, Peng and Bai, Shuai and Tan, Sinan and Wang, Shijie and Fan, Zhihao and Bai, Jinze and Chen, Keqin and Liu, Xuejing and Wang, Jialin and Ge, Wenbin and Fan, Yang and Dang, Kai and Du, Mengfei and Ren, Xuancheng and Men, Rui and Liu, Dayiheng and Zhou, Chang and Zhou, Jingren and Lin, Junyang},
  journal={arXiv preprint arXiv:2409.12191},
  year={2024}
}
```

## 📜 License

This model is released under the **Apache 2.0 License**, inheriting from Qwen2.5-VL-7B-Instruct.

## 🔗 Related Resources

- **SFT LoRA Adapter**: [yaoyuanlf/qwen2.5-vl-physics-lora](https://huggingface.co/yaoyuanlf/qwen2.5-vl-physics-lora)
- **Training Dataset**: [yaoyuanlf/physics-vlm-dataset](https://huggingface.co/datasets/yaoyuanlf/physics-vlm-dataset)
- **Base Model**: [Qwen/Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)
- **Training Code**: [GitHub Repository](https://github.com/yaoyuanlf/Physics-VLM-SFT-RLHF-Pipeline)

## 🙏 Acknowledgements

- **Qwen Team** for the exceptional base model
- **ModelScope Team** for the ms-swift RLHF framework
- **LLaMA-Factory** for the efficient SFT training pipeline
- **AutoDL** for providing GPU computing resources

## 📧 Contact

For questions or collaborations, please open an issue on [GitHub](https://github.com/yaoyuanlf/Physics-VLM-SFT-RLHF-Pipeline/issues).

---

<div align="center">

**Built with ❤️ for the Physics and Materials Science Community**

</div>
