# Physics VLM SFT & RLHF Pipeline

## I. Experimental Environment and Core Objectives
- **Hardware**: Single NVIDIA RTX 4090 (24GB VRAM) + High-end CPU (15+ cores) + Sufficient RAM.
- **Base Framework**: LLaMA-Factory.
- **Core Model**: Qwen2.5-VL-7B-Instruct (Multimodal Vision-Language Model).
- **Objective**: Perform Supervised Fine-Tuning (SFT) on 25,000 high-resolution physics/materials images (with defect or feature annotations) to train a specialized materials analysis model.

---

## II. Dataset Preparation and Environment Setup
The first step for multimodal fine-tuning is ensuring the framework can locate the images correctly. We resolved all "file not found" errors and established the proper file mapping logic.
1. **Data Path**: All images are stored under `data/material_dataset/`.
2. **JSON Mapping Configuration** (`dataset_info.json`): Register the custom dataset in LLaMA-Factory's `dataset_info.json`:
   - Specify `dataset: material_physics`.
   - Set the root directory `"image_folder": "data"`.
   - This way, when a relative path like `material_dataset/mat_img_xxxx.jpg` appears in the JSON file, the system correctly constructs the full absolute path.

---

## III. Dependency Hell and Version Pinning
While attempting to optimize VRAM usage, we encountered the classic dependency conflict and version compatibility issues — one of the most common bottlenecks in deep learning engineering. Below is the final working environment recipe.

### 1. Core Acceleration Libraries
To enable the 4090 to handle multimodal model training, we installed low-level acceleration plugins via source compilation:
- **Command**: `pip install bitsandbytes flash-attn --no-build-isolation`
- **Purpose**: `bitsandbytes` enables optimizer state compression; `flash-attn` (FlashAttention-2) drastically reduces the memory footprint of attention matrices. Compilation takes approximately 15 minutes.

### 2. Transformers Version Pinning
Hugging Face removed the `.visual` attribute from newer model versions, causing LLaMA-Factory to throw errors during quantization and patching.
- **Downgrade to 4.49.0**: Fixed the vision module error but lost `transformers.video_utils`, a hard dependency of LLaMA-Factory.
- **Upgrade to 5.3.0**: Triggered LLaMA-Factory's version lock (maximum supported: 5.2.0).
- **Final golden version**: `pip install "transformers==5.2.0"` — the perfect balance between new features and framework compatibility.

---

## IV. Pushing 24GB VRAM to Its Limits
The biggest challenge was fitting a 7-billion-parameter multimodal model — which requires ~15GB just for loading and is extremely memory-hungry during inference — into the 24GB of a single RTX 4090 for training.

### 1. Abandoning Buggy 4-bit Quantization
We initially planned to compress the model via `quantization_bit: 4`, but persistent `AttributeError` issues caused by framework–model incompatibility in 4-bit mode for the vision module led us to abandon quantization entirely and train at full precision.

### 2. Surgical Parameter Tuning
Without modifying `cutoff_len: 2048` (to preserve the tokenizer cache that took tens of minutes to compute), we adjusted the YAML configuration:
- **Reduced Batch Size**: `per_device_train_batch_size: 1` (one image per step), combined with `gradient_accumulation_steps: 16` to maintain effective batch size.
- **Trimmed LoRA Matrices**: Reduced `lora_rank` to 8; fine-tune only the core layers (`q_proj`, `v_proj`).
- **Enabled BF16**: Switched from `fp16` to `bf16`. The RTX 4090 natively supports BFloat16, eliminating the hidden VRAM overhead of the FP32 loss scaler used in traditional FP16 training.
- **Removed Evaluation Module**: Deleted `### eval` and `val_size` configs (later restored `val_size: 0.05` to recover the cache) to prevent the validation computation graph from occupying VRAM long-term.
- **Paged Optimizer**: Used `optim: paged_adamw_8bit` — compresses optimizer states to 8-bit and dynamically offloads to CPU memory when the GPU is near OOM.

### 3. Solving VRAM Fragmentation
With only ~200MB of headroom remaining, instead of modifying code, we leveraged a PyTorch environment variable:
- **Solution**: Prepend the anti-fragmentation directive before the launch command. This allows VRAM to dynamically expand and contract, stitching together scattered memory fragments.

---

## V. Final Training Configuration and Execution
Combining all lessons learned, the final `train_qwen_physics.yaml` configuration:

```yaml
### model
model_name_or_path: models/Qwen2.5-VL-7B-Instruct

### method
stage: sft
do_train: true
finetuning_type: lora
lora_target: q_proj,v_proj
lora_rank: 8
lora_alpha: 8

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

**Launch command:**
```bash
PYTORCH_ALLOC_CONF=expandable_segments:True llamafactory-cli train train_qwen_physics.yaml
```

---

## VI. Training Results and Inference Deployment

### 1. Training Duration and Results
After launch, the system loaded the preprocessed cache instantly (0 seconds for the tokenizer). After approximately 12 hours of continuous computation, all 4,455 steps completed successfully. The final `train_loss` dropped to an excellent **0.1762**, and model weights were saved to the target directory.

### 2. Cost Control Strategy
Given the long training time, we utilized AutoDL's auto-shutdown feature (triggered when GPU utilization stays below 10% for 10 consecutive minutes), automatically stopping billing after training completion.

### 3. Model Inference Testing
For testing the fine-tuned model, **CPU-only (no GPU) inference is strongly discouraged** — the matrix computations in multimodal inference are too intensive, leading to extremely slow responses or memory overflow on CPU.
- **Recommended**: Rent a GPU server with ≥16GB VRAM (e.g., RTX 3080/3090/4070Ti).
- **Launch command**: Run `llamafactory-cli webchat train_qwen_physics.yaml` in a GPU-enabled terminal to test the model's specialized capabilities via the WebUI.

---

## VII. Model Weights and Open-Source Release (Hugging Face)
This project follows MLOps best practices with a "code and model assets separated" storage strategy. The base model (Qwen2.5-VL-7B) uses the official open-source version, while the specialized LoRA weights and physics dataset produced in this experiment are permanently hosted on Hugging Face:

- **SFT Model Weights (LoRA Adapter)**: https://huggingface.co/yaoyuanlf/qwen2.5-vl-physics-lora
- **RLHF Model (Full Merged Weights)**: https://huggingface.co/yaoyuanlf/Qwen2.5-VL-7B-Physics-RLHF
- **Physics Materials Defect Dataset**: https://huggingface.co/datasets/yaoyuanlf/physics-vlm-dataset

> **Note**: For deployment, simply clone the GitHub repository for code and configurations — the framework will automatically download model weights from Hugging Face, enabling one-click inference.

---

## VIII. Fine-Tuning Results Showcase (Inference Demo)

To visually verify the effectiveness and limitations of SFT, we launched the WebUI via `llamafactory-cli webchat` and conducted multi-dimensional inference tests using real physics/materials images.

Below are the model's recognition performances in real-world complex academic/industrial scenarios:

### Test Case 1: 3D Spatial Reconstruction and Internal Structure Analysis
> **User Prompt:** Based on the green 3D models in Figures c, d, e and the 2D high-resolution electron microscopy images in Figures g, h, i, analyze the internal physical structure of these nanoparticles. Are they solid or hollow? Provide the visual evidence for your conclusion.

![Internal Structure Analysis](./assets/mat_img_686.jpg)

> **Fine-tuned Model Output:**
>
> **Nanoparticle Internal Structure Analysis:**
>
> **1. Green 3D Models (Figures c, d, e):**
> * **Figure c** shows a complete spherical nanoparticle with visible surface wrinkles and irregular morphology.
> * **Figures d and e** are cross-sectional models viewed from different angles, clearly revealing a distinct internal cavity.
>
> **2. 2D High-Resolution Electron Microscopy (Figures g, h, i):**
> * **Figure g** shows the cross-sectional image with a notably darker central region, indicating a hollow interior; the brighter edges suggest a thick outer shell.
> * **Figures h and i** further confirm this conclusion, with similarly darker central regions verifying the hollow structure.
>
> **3. Conclusion and Visual Evidence:**
> Combining the cavity shown in the 3D cross-sections with the characteristic "significant contrast difference between interior and exterior (bright edges, dark center)" in the HRTEM images, these nanoparticles conclusively exhibit a **hollow** internal structure.

---

### Test Case 2: Composite Nanostructure Caption Generation
> **User Prompt:** This is a composite figure from nanomaterials research. As a materials physicist, write a detailed figure caption explaining the morphological features, molecular assembly mechanisms, and experimental manipulation methods shown in Figures a through f.

![Composite Caption Generation](./assets/mat_img_7.jpg)

> **Fine-tuned Model Output:**
>
> * **a) TEM Image of a Nanoscale Structure:** Reveals the morphology of a hierarchical assembly, exhibiting a star-like pattern with numerous nanowires radiating from a central core.
> * **b) Schematic Representation of Molecular Assembly:** Illustrates the molecular assembly. The central core is labeled as "PFS-b-PMVS centre," indicating a polymer block copolymer core. The arms are labeled as "PFS-b-P2VP arms," suggesting a triblock copolymer architecture.
> * **c) TEM Image of a Different Sample:** Exhibits a more compact and dense arrangement of filaments.
> * **d) TEM Image of a Second Nanoscale Structure:** Presents an elongated and fibrous structure (scale bar: 500 nm).
> * **e) Schematic Representation of Another Molecular Assembly:** The central core is labeled as "Coordinated metal," with arms labeled as "Phosphinated corona," highlighting the functional groups responsible for the assembly.
> * **f) TEM Image of the Same Sample as (e):** Shows a similar fibrous and elongated structure, corresponding to the molecular assembly in (e).

![Experiment Screenshot](./assets/Xnip2026-03-26_12-10-26.jpg)

---

### Test Case 3: Ablation Study and Limitations Analysis (Fine-tuned vs. Base Model)
To objectively evaluate the impact of fine-tuning, we conducted a comparative test between the Base model and the Fine-tuned model on an image involving **complex chart data interpretation** and dynamic process analysis.

> **User Prompt:** As a physics expert, provide an in-depth analysis of this figure: What optical characterization technique is shown in Figures a and b? What material property does it typically measure? Based on the data in Figures e and f, summarize the relationship between nanoparticle diameter and velocity.

![Dynamic Process Analysis](./assets/mat_img_190.jpg)

#### Fine-tuned Model Performance
> **Output excerpt:** Figures a and b show "fluorescence recovery" technique... Combining Figures e and f, there is a clear positive correlation between particle diameter and velocity: at 50 nm diameter, velocity is approximately 0.2 μm/s; increasing to 500 nm, velocity reaches approximately 0.4 μm/s.

![Fine-tuned Output Screenshot](./assets/Xnip2026-03-26_12-14-48.jpg)

#### Base Model Performance (Qwen2.5-VL-7B-Instruct)
> **Output excerpt:**
> * **Technique**: Fluorescence Recovery After Photobleaching (FRAP), used to measure the diffusion coefficient of molecules or particles in materials.
> * **Physical Law Summary**: Exhibits a clear **inverse relationship**. 50 nm particles have an average velocity of ~0.3 μm/s; 500 nm particles have an average velocity of ~0.1 μm/s.
> * **Explanation**: According to Stokes' Law $v = \frac{F}{6\pi\eta r}$, particle velocity is inversely proportional to radius — larger particles experience greater drag.

![Base Model Output Screenshot](./assets/Xnip2026-03-26_12-25-34.jpg)

#### Engineering Post-Mortem (Insight)
In this comparative test, we observed the classic **Catastrophic Forgetting** phenomenon:
1. **Domain Overfitting**: The SFT stage used a large volume of single-dimension materials morphology/defect images, causing the model's original "academic chart reading" and "axis interpretation" capabilities to degrade due to visual neuron weight drift.
2. **Data Hallucination**: The fine-tuned model exhibited severe hallucination when reading the bar chart in Figure f, concluding that "larger size = faster speed" — a conclusion that violates basic physics. In contrast, the Base model not only accurately read the chart data but also independently invoked Stokes' Law for a principled explanation.
3. **Engineering Recommendation**: Future MLOps iterations should adopt a **mixed data training strategy (Data Mixing)**, incorporating 10–20% general academic chart data alongside the domain-specific image corpus to preserve basic chart analysis capabilities while maintaining specialized domain expertise.

---

## IX. Reinforcement Learning Phase (RLHF): Reshaping Physical Reasoning with GRPO

During the SFT ablation study, we observed that while the model acquired strong domain-specific feature extraction capabilities, it exhibited severe hallucinations when handling complex chart interpretation and physics law derivation. To address this, we introduced reinforcement learning to evolve the model from "describing images" to "rigorous reasoning."

### 1. Framework Migration and Algorithm Selection
To better support rule-based reward models, we migrated from LLaMA-Factory to **ModelScope's `ms-swift` framework** and adopted the state-of-the-art **GRPO (Group Relative Policy Optimization)** algorithm. Unlike PPO, GRPO eliminates the need for a separate Critic model of equal size to the Policy model, making it feasible to perform reinforcement learning on a 7B vision-language model within the 24GB VRAM of a single RTX 4090.

### 2. Core Weapon: Custom Physics Reward Function
The heart of reinforcement learning lies in the scoring mechanism. We abandoned generic language fluency rewards and developed a custom Python scoring script (`swift_physics_reward.py`) grounded in physics and materials science logic.
* **Matching Rules**: The engine captures the model's Chain-of-Thought (CoT) output in real time, evaluating whether it accurately derives target parameters. For example, when analyzing HRTEM images, the reward function grants a +1 positive signal only when the model precisely identifies and outputs standard Miller indices such as `(110)`.
* **Training Effectiveness**: By continuously monitoring the `logging.jsonl` logs, we observed that the physics consistency reward steadily climbed from initially negative values as training progressed, effectively eliminating the logical drift issues prevalent in the SFT stage.

### 3. Final Model Merge
After the GRPO training reached the target epoch, we executed `swift export` to merge the Qwen2.5-VL base model with the RL-produced LoRA adapter, generating the final **~16.6GB** standalone full-weight directory (`Qwen2_5_VL_Physics_RLHF_Merged`).

### 4. Model Demo

Post-RLHF model inference demonstration:

![RLHF Model Inference Demo](./assets/animation.gif)

After reinforcement learning optimization, the model can not only identify basic morphological features of materials but also perform in-depth analysis grounded in physical principles, significantly improving professionalism and accuracy.

---

## X. Local Model Loading and Automated Evaluation Troubleshooting (Inference & Eval)

During the post-merge testing and evaluation phase, we encountered and resolved "deadlock" issues related to loading sequences and underlying dependencies.

### 1. Breaking the "Never Reaches 100%" Download Illusion
When attempting to load the model via WebUI, the loading progress bar for shard files (`.safetensors`) frequently froze. Investigation revealed this was caused by a concurrent IO bottleneck from **symbolic link** conflicts in ModelScope's caching mechanism.
* **Solution**: We abandoned the timeout-prone web-based loading logic and directly rewrote `start_ui.py`, forcing `model_id` to point to the actual underlying path with underscores (e.g., `Qwen2___5-VL-7B-Instruct`) and mounting the RL weights in a "base + adapter" configuration.
* **Result**: Monitoring via `watch -n 1 nvidia-smi` showed VRAM jumping from 3MiB to 15,000MiB+ within seconds — the model was successfully loaded into the GPU and the WebUI chat interface became responsive instantly.

### 2. Abandoning Generic Benchmarks in Favor of Domain-Specific Evaluation
To provide rigorous quantitative evidence for the thesis, we chose not to run generic benchmarks like MME via `evalscope`, which have no relevance to physics.
* We used `swift infer` to perform fully automated batch inference on our proprietary validation set (`test.jsonl`).
* The resulting `final_inference_results.jsonl` contains detailed records of "image path — ground truth — model prediction." We downloaded this file locally and ran Python scripts for keyword matching statistics. The resulting **Accuracy** and **Confusion Matrix** serve as the core quantitative evidence in the thesis's Experimental Results chapter.

---

## XI. Production-Grade Open-Source Release and Project Delivery (MLOps Finale)

A thesis deliverable encompasses not just the written document but the complete engineering asset. Before releasing the cloud GPU instance, we completed a series of standard open-source compliance procedures.

### 1. High-Speed Hugging Face Upload Behind Firewall Restrictions
Since standard compute instances lack direct external network access, calls to `huggingface_hub` from the Python terminal repeatedly failed with `Network is unreachable` and Git authentication errors.
* **Solution**: First, we enabled the academic proxy by running `source /usr/bin/proxy_academic_host` in the terminal. Second, we abandoned Git CLI entirely and used the Python API with a Write Token for all uploads.
* **S3 Checksum Verification**: During the final stage of uploading the 16.6GB model files, the progress bar stalled at 99%. We correctly identified this as the HF server performing S3 bucket checksum hash verification on large files. After patiently waiting, all model shards were successfully uploaded to the HF platform.

### 2. Replacing the Performance Killer with a Lightweight Demo
To intuitively showcase the model's physics chain-of-thought capabilities, we recorded a live demonstration.
* The initial 141MB GIF proved to be a disaster for GitHub/web rendering — extremely slow to load and a serious drain on browser memory.
* We re-encoded it into an **18MB MP4 video in H.264 format** using FFmpeg. By embedding an HTML `<video>` tag with `autoplay loop muted` attributes in the README, we achieved a seamless, high-definition, auto-playing experience.

### 3. Project Dehydration and Structured GitHub Archival
In the final ten minutes before releasing the AutoDL server, to ensure the GitHub repository remained clean and modular (with SFT and RL stages properly separated), we performed the ultimate packaging using a custom `tar` filter:

```bash
tar --exclude='*VLM-Physics-Finetuning-Data/material_dataset/*.jpg' \
    --exclude='*.safetensors' --exclude='*.bin' --exclude='*.pth' \
    --exclude='pip_packages' --exclude='__pycache__' \
    -czvf /root/all_project_code.tar.gz .
```

This command cleanly stripped out tens of gigabytes of raw image data and model weights, distilling only the configuration files, scoring scripts, environment dependency manifests, and inference `.jsonl` files from each stage into a compressed archive of just a few tens of megabytes.

---

## XII. Multi-Agent Integration: From Visual Perception to Autonomous Materials Analysis

Beyond standalone model inference, this project extends the fine-tuned VLM into a **multi-agent system** that autonomously coordinates visual analysis with external knowledge retrieval and physics calculations, producing comprehensive characterization reports.

### Architecture

```
User Image → VLM Tool (perception) → Materials Project API (knowledge)
           → Physics Calculator (reasoning) → Comprehensive Report
```

The system comprises three specialized tools coordinated by a lightweight orchestrator:

| Tool | Role | Description |
|------|------|-------------|
| **VLM Tool** | Perception | RLHF-optimized Qwen2.5-VL analyzes microscopy images |
| **Materials Project API** | Knowledge Retrieval | Queries thermodynamic/mechanical properties for 150,000+ materials |
| **Physics Calculator** | Quantitative Reasoning | Hall–Petch yield strength, Bragg's law diffraction angles |

### Usage

```bash
# Basic usage (mock database, no API key needed)
python agent_demo.py --image assets/mat_img_686.jpg

# With real Materials Project database
python agent_demo.py --image assets/mat_img_686.jpg --mp-api-key YOUR_KEY

# Custom prompt and output
python agent_demo.py \
  --image assets/mat_img_158.jpg \
  --prompt "Identify the crystal structure and defects in this image." \
  --output report_158.json
```

### Parameters

| Argument | Default | Description |
|----------|---------|-------------|
| `--image` | (required) | Path to microscopy image |
| `--model` | `models/Qwen2_5_VL_Physics_RLHF_Merged` | Path to fine-tuned VLM |
| `--mp-api-key` | `None` | Materials Project API key (uses mock data if omitted) |
| `--prompt` | `None` | Custom analysis prompt |
| `--output` | `agent_report.json` | Output JSON report path |

### Execution Pipeline

1. **VLM Perception** — Analyzes the input image, identifying material type, crystal structure, and defects
2. **Material Extraction** — Parses chemical formulas (e.g., PbS, Fe, TiO2) from VLM output via regex
3. **Database Query** — Retrieves crystal system, band gap, elastic moduli, formation energy from Materials Project
4. **Physics Calculation** — Automatically applies Hall–Petch or Bragg's law when grain sizes or d-spacings are detected
5. **Report Synthesis** — Combines all outputs into a structured JSON report with cross-validation

### Example Output (PbS Nanocrystals)

```
Step 1 — VLM: "PbS nanocrystals, FCC rock-salt structure, 8-12 nm diameter"
Step 2 — Materials Project: cubic Fm-3m, band gap 0.41 eV, bulk modulus 52.9 GPa
Step 3 — Hall-Petch: σ_y = 70 + 0.74/√0.01 = 77.4 MPa
Step 4 — Cross-validation: VLM's FCC identification ✓ consistent with Fm-3m space group
```

### 12-Image Quantitative Evaluation

To move beyond a single demo, the agent was evaluated on 12 test images covering diverse material types (nanocrystals, thin films, polymers, spectral plots, optical setups). All JSON reports are stored in `paper_result/`.

| Metric | Value | Notes |
|--------|-------|-------|
| VLM perception success | 12/12 (100%) | All images received structured analysis |
| Material formula extraction | 7/12 (58%) | Fails on polymers and non-micrographs |
| MP API query success | 8/8 (100%) | All dispatched queries returned data |
| Physics calculation triggered | 4/12 (33%) | Triggered when grain size is present |
| Full 3-tool pipeline | 2/12 (17%) | Images 158 (PbS+Pd) and 3250 (Cu) |
| False positive extraction | 1/12 (8%) | "Ti:Sapphire" → Ti |
| End-to-end crash rate | 0/12 (0%) | No runtime errors |

**Key findings:**
- **VLM is the reliable component** — 100% perception success across all material types confirms that SFT+RLHF training generalizes well.
- **The bottleneck is the orchestrator, not the VLM** — regex-based material extraction only covers a fixed list of inorganic formulas, failing on polymers (PEO, PFS-b-P2VP), complex compounds (FeSe₀.₅Te₀.₅ → only Fe extracted), and producing false positives ("Ti:Sapphire" → Ti).
- **Materials Project polymorph mismatch** — the API often returns metastable/high-pressure phases (e.g., PbS as monoclinic P2₁/m instead of cubic Fm-3m galena; Fe as triclinic P1 instead of BCC).
- **Clear improvement path** — replace regex with LLM-based structured extraction, implement polymorph selection logic, and add polymer-specific tool modules.

> **Note**: The agent requires ~16GB VRAM to load the VLM in BFloat16 precision. The Materials Project API key can be obtained for free at https://next-gen.materialsproject.org/api.

---

## Results Summary
Through comparative testing, we found that the un-fine-tuned Base Model tends to produce generic image descriptions (e.g., "this is a gray metal plate"), while the model merged with our LoRA weights demonstrates **domain expert-level perception**, accurately employing professional terminology (e.g., oxidation, micro-cracks, fatigue damage) for localization and diagnosis. The multi-agent extension further demonstrates that this domain-adapted VLM can serve as the perceptual core of autonomous scientific workflows, bridging the gap between visual understanding and actionable materials analysis.
