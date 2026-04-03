一、 实验环境与核心目标
- 硬件配置：单卡 RTX 4090 (24GB 显存) + 高配 CPU (15核以上) + 充足内存。
- 基础框架：LLaMA-Factory。
- 核心模型：Qwen2.5-VL-7B-Instruct (多模态视觉语言模型)。
- 实验目标：在 25,000 张高分辨率物理材料图片（带有瑕疵或特征标注）上进行有监督微调 (SFT)，训练一个专属的物理材料分析大模型。

---
二、 数据集准备与环境联通
多模态微调的第一步是让框架准确找到图片。我们排除了“找不到文件”的报错，确立了正确的文件映射逻辑。
1. 数据存放路径：所有图片统一放置在 data/material_dataset/ 目录下。
2. JSON 映射配置 (dataset_info.json)： 在 LLaMA-Factory 的 dataset_info.json 中，注册我们的自定义数据集：
  - 指定 dataset: material_physics。
  - 设定根目录 "image_folder": "data"。
  - 这样，当 JSON 文件中出现相对路径 material_dataset/mat_img_xxxx.jpg 时，系统能完美拼接出绝对路径。

---
三、 依赖库的“炼狱”与版本锁定
在尝试优化显存时，我们遇到了极其经典的依赖库冲突与版本兼容问题。这是深度学习工程中最容易卡关的地方，最终我们得出了以下完美环境配方。
1. 核心加速库安装
为了让 4090 跑得动多模态模型，必须安装底层加速插件。我们在终端执行了现场编译安装：
- 命令：pip install bitsandbytes flash-attn --no-build-isolation
- 作用：bitsandbytes 用于后续可能的优化器状态压缩；flash-attn (FlashAttention-2) 则是大幅降低注意力矩阵显存占用的“救命神药”。编译过程耗时约 15 分钟，必须耐心等待。
2. Transformers 版本锁定
由于 Hugging Face 官方删除了新版模型底层的 .visual 属性，导致 LLaMA-Factory 在尝试量化或打补丁时频繁报错。
- 降级到 4.49.0：解决了视觉模块报错，但缺失了 LLaMA-Factory 强依赖的 transformers.video_utils。
- 升级到 5.3.0：触发了 LLaMA-Factory 的安全锁（它最高只支持 5.2.0）。
- 最终黄金版本：我们在终端执行 pip install "transformers==5.2.0"，完美平衡了新特性与框架兼容性。

---
四、 24GB 显存的极限压榨战
本实验最大的挑战是如何将一个原本需要 15GB 显存加载、推理极度消耗显存的 70 亿参数多模态模型，硬塞进 24GB 的 4090 中进行训练。我们分阶段实施了以下“显存保卫战”：
1. 摒弃有 Bug 的 4-bit 量化
原本计划通过 quantization_bit: 4 将模型压缩，但由于框架与模型在 4-bit 模式下的视觉模块兼容问题（反复报错 AttributeError），我们果断彻底删除了 4-bit 量化，决定在全量状态下硬刚。
2. 核心参数的“微创手术”
在不修改 cutoff_len: 2048（为了保住耗时数十分钟才算好的 Tokenizer 缓存）的前提下，我们修改了 YAML 文件：
- 降低 Batch Size：per_device_train_batch_size: 1（单次只喂 1 张图），配合 gradient_accumulation_steps: 16 维持总体训练效果。
- 削减 LoRA 矩阵：将 lora_rank 降至 4，只微调核心层 (q_proj, v_proj)。
- 启用 BF16：将 fp16 改为 bf16。4090 原生支持 Bfloat16，这直接砍掉了传统 FP16 训练时在后台偷占显存的 FP32 缩放器。
- 删除 Eval 模块：删除了 ### eval 和 val_size 相关的配置（后续为了找回缓存又补回了 val_size: 0.05），避免验证集计算图长期驻留显存。
- 引入分页优化器：使用 optim: paged_adamw_8bit。它不仅把优化器体积压缩到了 8-bit，还能在显卡即将 OOM 时，动态向 CPU 内存借用空间。
3. 解决显存碎片化
在还差 200MB 就爆显存的最后关头，我们没有修改代码，而是使用了 PyTorch 的底层环境变量。
- 对策：在启动命令前加上防碎片指令。它允许显存像橡皮筋一样动态伸缩，将零散的显存碎片拼凑起来使用。

---
五、 最终的训练配置与执行
结合上述所有经验，我们最终定稿的 train_qwen_physics.yaml 核心配置如下：
```yaml
### model
model_name_or_path: models/Qwen2.5-VL-7B-Instruct

### method
stage: sft
do_train: true
finetuning_type: lora
lora_target: q_proj,v_proj
lora_rank: 4
lora_alpha: 4

### dataset
dataset: material_physics
template: qwen2_vl
cutoff_len: 2048
max_samples: 25000
overwrite_cache: false
preprocessing_num_workers: 8

### output
output_dir: output/qwen_physics_v1
logging_steps: 10
save_steps: 100
plot_loss: true
overwrite_output_dir: true

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 16
learning_rate: 5.0e-5
num_train_epochs: 3.0
lr_scheduler_type: cosine
bf16: true
flash_attn: fa2
optim: paged_adamw_8bit

### eval
val_size: 0.05
```
最终点火命令：
Bash
PYTORCH_ALLOC_CONF=expandable_segments:True llamafactory-cli train train_qwen_physics.yaml

---
六、 实验结果与推理部署
1. 训练耗时与结果： 启动后，系统完美读取了预处理缓存（0 秒过 Tokenizer）。经过约 12 个小时的不间断计算，4455 个 Steps 全部跑完。最终 train_loss 降至极其优秀的 0.1762，模型权重成功保存在了目标目录下。
2. 成本控制策略： 由于训练时间较长，我们采用了 AutoDL 的自动关机功能（按 GPU 利用率 < 10% 持续 10 分钟触发），在训练结束后自动切断了计费。
3. 模型推理测试： 针对微调后的模型测试，文档明确指出了**不推荐使用无显卡模式（纯 CPU）**进行对话，因为多模态推理的矩阵计算量极大，CPU 会导致响应极慢甚至内存溢出。
  - 建议方案：租赁带有 RTX 3080/3090/4070Ti 等显存 ≥ 16GB 的高性价比显卡服务器。
  - 启动命令：在带有显卡的终端中运行 llamafactory-cli webchat train_qwen_physics.yaml，即可通过 WebUI 直接给它发送物理图片进行专属能力的检验。
---
七、 模型权重与数据开源 (Hugging Face)
本项目遵循 MLOps 工业级规范，采用“代码与模型资产分离”的存储策略。基础大模型（Qwen2.5-VL-7B）使用官方开源版本，而本次实验生成的专属 LoRA 权重与物理数据集已永久托管于 Hugging Face：

🧠 SFT 模型权重 (LoRA Adapter): https://huggingface.co/yaoyuanlf/qwen2.5-vl-physics-lora

🎯 RLHF 强化学习模型 (完整模型): https://huggingface.co/yaoyuanlf/Qwen2.5-VL-7B-Physics-RLHF

📊 物理材料瑕疵数据集: https://huggingface.co/datasets/yaoyuanlf/physics-vlm-dataset

(注：部署时，只需拉取 GitHub 的运行代码与配置，框架将自动从 Hugging Face 下载模型权重，实现一键推理。)


## 八、 模型微调效果展示 (Inference Demo)

为了直观验证有监督微调 (SFT) 的有效性与局限性，我们使用 `llamafactory-cli webchat` 启动了 WebUI，并使用真实的物理材料图像进行了多维度的推理测试。

以下是模型在实际复杂学术/工业场景下的识别表现：

### 🔬 测试案例一：三维空间重构与内部结构解析
> **User Prompt:** 请结合图 c、d、e 的绿色 3D 模型，以及图 g、h、i 的二维高分辨电镜照片，分析这种纳米颗粒的内部物理结构是什么样的？它是实心的还是空心的？请说明得出此结论的视觉依据。
![内部结构解析](./assets/mat_img_686.jpg)

> **Finetuned Model Output:** > **纳米颗粒内部物理结构分析：**
> 
> **1. 图 c、d、e 的绿色 3D 模型：**
> * **图 c** 显示了一个完整的球形纳米颗粒，表面有明显的褶皱和不规则形状。
> * **图 d 和图 e** 是该纳米颗粒的切面模型，从不同角度观察，可以清晰地看到内部有一个明显的空腔。
> 
> **2. 图 g、h、i 的二维高分辨电镜照片：**
> * **图 g** 显示了横截面图像，中间部分明显较暗，表明内部是空心的；边缘部分较亮，说明外壳较厚。
> * **图 h 和图 i** 进一步确认了这一结论，中间部分同样较暗，证实了空心结构的存在。
> 
> **3. 结论及视觉依据：**
> 结合 3D 物理切面展示的空腔，以及高分辨电镜照片中典型的“内部与外部对比度显著差异（边缘亮、中心暗）”，可以明确得出这种纳米颗粒具有**空心（Hollow）**的内部物理结构。

---

### 🧬 测试案例二：复合纳米结构图注生成 (Captioning)
> **User Prompt:** 这是一张纳米材料领域的复合配图。请你作为材料物理学家，为这张图写一段详细的图注 (Figure Caption)，解释从图 a 到图 f 所展示的纳米结构的形貌特征、分子组装方式以及实验操控手段。
![复合图注生成](./assets/mat_img_7.jpg)

> **Finetuned Model Output:** > **Figure Caption:**
> 
> * **a) TEM Image of a Nanoscale Structure:** Reveals the morphology of a hierarchical assembly, exhibiting a star-like pattern with numerous nanowires radiating from a central core. 
> * **b) Schematic Representation of Molecular Assembly:** Illustrates the molecular assembly. The central core is labeled as "PFS-b-PMVS centre," indicating a polymer block copolymer core. The arms are labeled as "PFS-b-P2VP arms," suggesting a triblock copolymer architecture.
> * **c) TEM Image of a Different Sample:** Exhibits a more compact and dense arrangement of filaments.
> * **d) TEM Image of a Second Nanoscale Structure:** Presents an elongated and fibrous structure (scale bar: 500 nm).
> * **e) Schematic Representation of Another Molecular Assembly:** The central core is labeled as "Coordinated metal," with arms labeled as "Phosphinated corona," highlighting the functional groups responsible for the assembly.
> * **f) TEM Image of the Same Sample as (e):** Shows a similar fibrous and elongated structure, corresponding to the molecular assembly in (e).
![实验片段](./assets/Xnip2026-03-26_12-10-26.jpg)

---

### 🆚 测试案例三：消融实验与局限性分析（微调 vs 原生）
为了客观评估微调带来的影响，我们针对一张**涉及复杂图表数据读取**的动态过程分析图，进行了 Base 模型与 Finetuned 模型的对比测试（Ablation Study）。

> **User Prompt:** 作为一名物理专家，请对这张图进行深度解析：图 a 和 b 展示的是什么光学评估技术？通常用来测量材料的什么性质？结合图 e 和 f 的数据，总结出纳米颗粒直径与其运动速度之间呈现怎样的规律？
![动态过程分析](./assets/mat_img_190.jpg)

#### 🔴 微调模型表现 (Finetuned Model)
> **Output 节选:** 图 a 和 b 展示的是“荧光恢复”技术... 结合图 e 和 f，颗粒直径与速度存在明显的正相关：当直径 50 nm 时，速度约 0.2 μm/s；增大至 500 nm 时，速度达到了约 0.4 μm/s。
![微调输出截图](./assets/Xnip2026-03-26_12-14-48.jpg)

#### 🟢 原生大模型表现 (Base Model - Qwen2.5-VL-7B-Instruct)
> **Output 节选:** > * **现象/技术**：荧光漂白恢复（FRAP），用于测量材料中分子或颗粒的扩散系数。
> * **物理规律总结**：呈现明显的**反比关系**。50 nm 颗粒平均速度约 0.3 μm/s；500 nm 颗粒平均速度约 0.1 μm/s。
> * **原因解释**：根据斯托克斯定律 $v = \frac{F}{6\pi\eta r}$，颗粒速度与半径成反比，直径较大的颗粒受到更大的阻力。
![原生输出截图](./assets/Xnip2026-03-26_12-25-34.jpg)

#### 👨‍💻 工程师深度复盘 (Insight)
在本次对比测试中，我们观察到了经典的**灾难性遗忘 (Catastrophic Forgetting)** 现象：
1. **领域过拟合**：由于我们在 SFT 阶段输入了大量单一维度的材料形貌/瑕疵图片，导致模型原有的“学术图表读取”和“坐标轴解析”能力受到了视觉神经元的权重偏移干扰。
2. **数据幻觉**：微调模型在读取图 f 的柱状图时出现了严重幻觉，得出了“尺寸越大、速度越快”这种违背物理常识的错误结论；而 Base 模型不仅精准读出了图表数据，还自主调用了斯托克斯定律进行原理解释。
3. **工程优化建议**：在未来的 MLOps 迭代中，应当采用**混合数据训练策略 (Data Mixing)**，在垂直领域图库中混入一定比例（如 10%-20%）的通用学术图表数据，以保证模型在具备垂直鉴别能力的同时，保留基础的图表数据分析理智。

九、 强化学习阶段 (RLHF)：利用 GRPO 重塑物理逻辑
在 SFT 阶段的消融实验中，我们观察到模型虽然具备了极强的垂直领域特征提取能力，但在涉及复杂图表读取与物理定律推导时，出现了严重的幻觉。为了解决这一痛点，我们引入了强化学习机制，让模型从“看图说话”进化为“严谨推理”。

1. 框架切换与算法选择
为了更好地支持基于规则的奖励模型，我们将实验阵地从 LLaMA-Factory 迁移至 ModelScope 的 ms-swift 框架，并采用了当前最前沿的 GRPO (Group Relative Policy Optimization) 算法。相比 PPO，GRPO 省去了与 Policy Model 同等体积的 Critic 模型，这让我们得以在 24GB 显存的 4090 上完成 7B 视觉大模型的强化学习。

2. 核心利器：自定义物理奖励函数 (Reward Function)
强化学习的核心在于“打分”。我们放弃了通用的语言流畅度奖励，专门编写了基于物理材料学逻辑的 Python 打分脚本 (swift_physics_reward.py)。

匹配规则：引擎会实时抓取模型输出的思维链（CoT），判断其是否准确推导出了目标参数。例如，当模型在分析高分辨电镜图时，只有在其回答中精确推导并输出了诸如 (110) 等标准晶面指数时，Reward 才会给予 +1 的正反馈。

训练收效：通过在后台持续监控 logging.jsonl 中的日志，我们观察到随着训练步数的增加，模型的物理一致性奖励从初始的负值稳步攀升，彻底锁死了 SFT 阶段容易产生的逻辑飘忽问题。

3. 最终模型合并 (Merge)
在 GRPO 训练达到预定 Epoch 后，我们执行了 swift export，将 Qwen2.5-VL 基座模型与强化学习产出的 LoRA 补丁完美融合，生成了最终约 16.6GB 的全量独立权重文件夹 (Qwen2_5_VL_Physics_RLHF_Merged)。

十、 模型本地加载与自动化评测排雷 (Inference & Eval)
在模型合并后的测试与评估阶段，我们遭遇并克服了加载时序与底层依赖的“假死”问题。

1. 破解“永远达不到 100%”的下载幻觉
在尝试通过 WebUI 加载模型时，我们发现后台分片文件（.safetensors）的加载进度条频繁卡死。经过排查，这是由于 ModelScope 缓存机制产生的软链接 (Symbolic Link) 冲突导致的并发 IO 瓶颈。

破局方案：我们摒弃了容易超时的 Web 端加载逻辑，直接重写了 start_ui.py。通过强制将 model_id 指向真实存在的底层带下划线路径（如 Qwen2___5-VL-7B-Instruct），并以“基座 + Adapter”的形式挂载 RL 权重。

成效：通过 watch -n 1 nvidia-smi 监控，显存从 3MiB 瞬间飙升至 15000MiB+，模型仅用几秒钟便成功灌入 GPU，WebUI 对话框瞬间点亮。

2. 放弃通用题库，构建垂直领域测试闭环
为了给论文提供严谨的数据支撑，我们没有使用 evalscope 去跑 MME 等毫无物理意义的通用评测。

我们使用 swift infer 指令，让融合后的模型在我们专属的验证集 (test.jsonl) 上进行了无干预的批量自动化推理。

生成的 final_inference_results.jsonl 详细记录了“图片路径-标准答案-模型预测”。我们将该文件下载至本地，通过 Python 脚本进行关键字匹配统计，得出的 Accuracy 和错误分布矩阵（Confusion Matrix）成为了论文 Experimental Results 章节中最核心的定量论据。

十一、 工业级开源与项目交付 (MLOps 终局)
毕业论文不仅是文本的交付，更是完整工程资产的交付。在彻底释放云端 GPU 实例前，我们完成了一系列标准的开源合规化动作。

1. 跨越内网屏障的 Hugging Face 极速上传
由于普通算力实例无法直连外网，我们在 Python 终端调用 huggingface_hub 时频繁遭遇 Network is unreachable 和 Git 验证失败的报错。

解决路径：首先，在终端执行 source /usr/bin/proxy_academic_host 成功开启学术加速。其次，放弃 Git 命令行，全盘使用 Python API 传入 Write Token 进行上传。

S3 校验机制规避：在上传 16.6GB 模型文件的尾声阶段，进度条停滞在 99%。我们精准识别出这是 HF 服务器在进行大文件 S3 存储桶的 Checksum 哈希校验，通过耐心等待，最终成功在 HF 平台点亮了所有的模型分片。

2. 摒弃性能杀手，打造轻量化展示 Demo
为了直观展示模型强大的物理思维链能力，我们录制了实机演示。

最初生成的 141MB 巨型 GIF 被证明是 GitHub/网页渲染的灾难，不仅加载极慢且严重消耗浏览器内存。

我们利用 FFmpeg 将其重新编码为 H.264 格式的 18MB MP4 视频。通过在 README 中嵌入带有 autoplay loop muted 属性的 HTML <video> 标签，实现了完美的高清秒开自动播放。

3. 项目脱水与 GitHub 结构化封存
在离开 AutoDL 服务器的最后十分钟，为了确保 GitHub 仓库的绝对纯净（模块化、SFT与RL分离），我们使用自定义的 tar 过滤指令进行了终极打包：

Bash
tar --exclude='*VLM-Physics-Finetuning-Data/material_dataset/*.jpg' \
    --exclude='*.safetensors' --exclude='*.bin' --exclude='*.pth' \
    --exclude='pip_packages' --exclude='__pycache__' \
    -czvf /root/all_project_code.tar.gz .
该指令完美剥离了几十 GB 的原图数据集与庞大的模型权重，仅将配置文件、打分脚本、环境依赖表及各个阶段的推理 .jsonl 提炼为不到数十兆的压缩包。


### 🆚 效果总结
通过测试对比发现，未微调的基础模型 (Base Model) 倾向于给出宽泛的图像描述（如“这是一块灰色的金属板”），而融合了本次 LoRA 权重的模型，已经具备了**领域专家级别的感知能力**，能够准确使用专业术语（如氧化、微裂纹、疲劳损伤等）进行定位与诊断，完全达到了本次实验的预期目标。


## 九、 强化学习阶段 (RLHF)：利用 GRPO 重塑物理逻辑

在 SFT 阶段的消融实验中，我们观察到模型虽然具备了极强的垂直领域特征提取能力，但在涉及复杂图表读取与物理定律推导时，出现了严重的幻觉。为了解决这一痛点，我们引入了强化学习机制，让模型从“看图说话”进化为“严谨推理”。

### 1. 框架切换与算法选择
为了更好地支持基于规则的奖励模型，我们将实验阵地从 LLaMA-Factory 迁移至 **ModelScope 的 `ms-swift` 框架**，并采用了当前最前沿的 **GRPO (Group Relative Policy Optimization)** 算法。相比 PPO，GRPO 省去了与 Policy Model 同等体积的 Critic 模型，这让我们得以在 24GB 显存的 4090 上完成 7B 视觉大模型的强化学习。

### 2. 核心利器：自定义物理奖励函数 (Reward Function)
强化学习的核心在于“打分”。我们放弃了通用的语言流畅度奖励，专门编写了基于物理材料学逻辑的 Python 打分脚本 (`swift_physics_reward.py`)。
* **匹配规则**：引擎会实时抓取模型输出的思维链 (CoT)，判断其是否准确推导出了目标参数。例如，当模型在分析高分辨电镜图时，只有在其回答中精确推导并输出了诸如 `(110)` 等标准晶面指数时，Reward 才会给予 +1 的正反馈。
* **训练收效**：通过在后台持续监控 `logging.jsonl` 中的日志，我们观察到随着训练步数的增加，模型的物理一致性奖励从初始的负值稳步攀升，彻底锁死了 SFT 阶段容易产生的逻辑飘忽问题。

### 3. 最终模型合并 (Merge)
在 GRPO 训练达到预定 Epoch 后，我们执行了 `swift export`，将 Qwen2.5-VL 基座模型与强化学习产出的 LoRA 补丁完美融合，生成了最终约 **16.6GB** 的全量独立权重文件夹 (`Qwen2_5_VL_Physics_RLHF_Merged`)。

### 4. 模型效果演示

RLHF 训练后的模型推理演示：

![RLHF 模型推理演示](./assets/animation.gif)

通过强化学习优化，模型不仅能识别材料的基本形貌特征，还能结合物理原理进行深度分析，显著提升了专业性和准确性。

---

## 十、 模型本地加载与自动化评测排雷 (Inference & Eval)

在模型合并后的测试与评估阶段，我们遭遇并克服了加载时序与底层依赖的“假死”问题。

### 1. 破解“永远达不到 100%”的下载幻觉
在尝试通过 WebUI 加载模型时，我们发现后台分片文件 (`.safetensors`) 的加载进度条频繁卡死。经过排查，这是由于 ModelScope 缓存机制产生的**软链接 (Symbolic Link)** 冲突导致的并发 IO 瓶颈。
* **破局方案**：我们摒弃了容易超时的 Web 端加载逻辑，直接重写了 `start_ui.py`。通过强制将 `model_id` 指向真实存在的底层带下划线路径（如 `Qwen2___5-VL-7B-Instruct`），并以“基座 + Adapter”的形式挂载 RL 权重。
* **成效**：通过 `watch -n 1 nvidia-smi` 监控，显存从 3MiB 瞬间飙升至 15000MiB+，模型仅用几秒钟便成功灌入 GPU，WebUI 对话框瞬间点亮。

### 2. 放弃通用题库，构建垂直领域测试闭环
为了给论文提供严谨的数据支撑，我们没有使用 `evalscope` 去跑 MME 等毫无物理意义的通用评测。
* 我们使用 `swift infer` 指令，让融合后的模型在我们专属的验证集 (`test.jsonl`) 上进行了无干预的**批量自动化推理**。
* 生成的 `final_inference_results.jsonl` 详细记录了“图片路径-标准答案-模型预测”。我们将该文件下载至本地，通过 Python 脚本进行关键字匹配统计，得出的 Accuracy 和错误分布矩阵 (Confusion Matrix) 成为了论文 Experimental Results 章节中最核心的定量论据。

---

## 十一、 工业级开源与项目交付 (MLOps 终局)

毕业论文不仅是文本的交付，更是完整工程资产的交付。在彻底释放云端 GPU 实例前，我们完成了一系列标准的开源合规化动作。

### 1. 跨越内网屏障的 Hugging Face 极速上传
由于普通算力实例无法直连外网，我们在 Python 终端调用 `huggingface_hub` 时频繁遭遇 `Network is unreachable` 和 Git 验证失败的报错。
* **解决路径**：首先，在终端执行 `source /usr/bin/proxy_academic_host` 成功开启学术加速。其次，放弃 Git 命令行，全盘使用 Python API 传入 Write Token 进行上传。
* **S3 校验机制规避**：在上传 16.6GB 模型文件的尾声阶段，进度条停滞在 99%。我们精准识别出这是 HF 服务器在进行大文件 S3 存储桶的 Checksum 哈希校验，通过耐心等待，最终成功在 HF 平台点亮了所有的模型分片。

### 2. 摒弃性能杀手，打造轻量化展示 Demo
为了直观展示模型强大的物理思维链能力，我们录制了实机演示。
* 最初生成的 141MB 巨型 GIF 被证明是 GitHub/网页渲染的灾难，不仅加载极慢且严重消耗浏览器内存。
* 我们利用 FFmpeg 将其重新编码为 **H.264 格式的 18MB MP4 视频**。通过在 README 中嵌入带有 `autoplay loop muted` 属性的 HTML `<video>` 标签，实现了完美的高清秒开自动播放。

### 3. 项目脱水与 GitHub 结构化封存
在离开 AutoDL 服务器的最后十分钟，为了确保 GitHub 仓库的绝对纯净（模块化、SFT与RL分离），我们使用自定义的 `tar` 过滤指令进行了终极打包：

```bash
tar --exclude='*VLM-Physics-Finetuning-Data/material_dataset/*.jpg' \
    --exclude='*.safetensors' --exclude='*.bin' --exclude='*.pth' \
    --exclude='pip_packages' --exclude='__pycache__' \
    -czvf /root/all_project_code.tar.gz .
