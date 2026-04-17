"""
LLM-as-Judge Evaluation for Materials Science VLM
===================================================
Uses GPT-4o (or other LLM APIs) as an expert judge to evaluate
model outputs across multiple materials science dimensions.

This script:
1. Loads test images and runs inference with the fine-tuned VLM
2. Sends each (image, model_output) pair to GPT-4o for scoring
3. Aggregates scores across 5 dimensions and generates a report

Usage:
    # Full pipeline: VLM inference + GPT-4o judging
    python llm_judge_eval.py \
        --image-dir assets/ \
        --model models/Qwen2_5_VL_Physics_RLHF_Merged \
        --openai-key sk-xxxxx \
        --num-samples 30

    # Judge-only mode: evaluate pre-generated outputs
    python llm_judge_eval.py \
        --input-file inference_results.jsonl \
        --openai-key sk-xxxxx

    # Use Anthropic Claude instead of GPT-4o
    python llm_judge_eval.py \
        --image-dir assets/ \
        --model models/Qwen2_5_VL_Physics_RLHF_Merged \
        --anthropic-key sk-ant-xxxxx \
        --judge-model claude
"""

import argparse
import base64
import json
import os
import time
import statistics
from pathlib import Path


# ============================================================
# Judge Prompt Template
# ============================================================

JUDGE_PROMPT = """You are a senior materials scientist with expertise in electron microscopy, crystallography, and microstructural analysis. You are evaluating the quality of an AI model's analysis of a materials science microscopy image.

Below is the model's output for a given microscopy image. Please evaluate it on the following 5 dimensions, scoring each from 1 to 5:

**Scoring Criteria:**

1. **Material Identification (1-5):**
   - 1: Completely wrong or no identification
   - 3: Partially correct (e.g., correct material class but wrong specific phase)
   - 5: Precise identification with correct chemical formula and phase

2. **Physical Consistency (1-5):**
   - 1: Contains statements violating physical laws (e.g., wrong crystal system, impossible phase relationships)
   - 3: No obvious errors but lacks physical depth
   - 5: All statements physically correct with proper structure-property linkage

3. **Terminology & Professionalism (1-5):**
   - 1: Uses lay language, no domain vocabulary
   - 3: Some correct terms but inconsistent usage
   - 5: Consistently uses precise crystallographic and materials science terminology

4. **Observation Detail (1-5):**
   - 1: Vague, generic description applicable to any image
   - 3: Mentions some specific features visible in the image
   - 5: Detailed description referencing specific panels, scale bars, morphological features, and defects

5. **Reasoning Quality (1-5):**
   - 1: No reasoning, just listing observations
   - 3: Some cause-effect reasoning but superficial
   - 5: Multi-step reasoning connecting observations to mechanisms and implications

**Model Output:**
{model_output}

**Response Format (strict JSON only):**
{{
    "material_identification": <int 1-5>,
    "physical_consistency": <int 1-5>,
    "terminology": <int 1-5>,
    "observation_detail": <int 1-5>,
    "reasoning_quality": <int 1-5>,
    "justification": "<brief 2-3 sentence explanation>"
}}"""


# ============================================================
# VLM Inference
# ============================================================

def load_vlm(model_path):
    """Load the fine-tuned VLM for inference."""
    import torch
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

    print("[VLM] Loading model...")
    processor = AutoProcessor.from_pretrained(model_path)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto"
    )
    print("[VLM] Model loaded.")
    return model, processor


def vlm_inference(model, processor, image_path):
    """Run VLM inference on a single image."""
    import torch
    from qwen_vl_utils import process_vision_info

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": f"file://{Path(image_path).resolve()}"},
            {"type": "text", "text": "Please describe this material microstructure in detail."},
        ],
    }]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=512, temperature=0.7)

    generated = output_ids[0][inputs.input_ids.shape[1]:]
    return processor.decode(generated, skip_special_tokens=True)


# ============================================================
# LLM Judge (GPT-4o / Claude)
# ============================================================

def judge_with_openai(api_key, model_output, image_path=None, model_name="gpt-4o"):
    """Use OpenAI GPT-4o as judge."""
    import requests

    messages = [{"role": "user", "content": []}]

    # Optionally include image for visual grounding
    if image_path and os.path.exists(image_path):
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        ext = Path(image_path).suffix.lower()
        mime = "image/png" if ext == ".png" else "image/jpeg"
        messages[0]["content"].append({
            "type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{b64}", "detail": "low"}
        })

    messages[0]["content"].append({
        "type": "text",
        "text": JUDGE_PROMPT.format(model_output=model_output)
    })

    resp = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"model": model_name, "messages": messages, "temperature": 0.1, "max_tokens": 500},
        timeout=60,
    )

    if resp.status_code != 200:
        return {"error": f"API status {resp.status_code}: {resp.text[:200]}"}

    text = resp.json()["choices"][0]["message"]["content"]
    return _parse_scores(text)


def judge_with_anthropic(api_key, model_output, image_path=None, model_name="claude-sonnet-4-20250514"):
    """Use Anthropic Claude as judge."""
    import requests

    content = []

    if image_path and os.path.exists(image_path):
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        ext = Path(image_path).suffix.lower()
        mime = "image/png" if ext == ".png" else "image/jpeg"
        content.append({
            "type": "image",
            "source": {"type": "base64", "media_type": mime, "data": b64}
        })

    content.append({
        "type": "text",
        "text": JUDGE_PROMPT.format(model_output=model_output)
    })

    resp = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        },
        json={
            "model": model_name,
            "max_tokens": 500,
            "temperature": 0.1,
            "messages": [{"role": "user", "content": content}],
        },
        timeout=60,
    )

    if resp.status_code != 200:
        return {"error": f"API status {resp.status_code}: {resp.text[:200]}"}

    text = resp.json()["content"][0]["text"]
    return _parse_scores(text)


def _parse_scores(text):
    """Parse JSON scores from judge response."""
    import re
    # Extract JSON block
    match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if not match:
        return {"error": f"Could not parse JSON from: {text[:200]}"}
    try:
        scores = json.loads(match.group())
        # Validate all required fields
        required = ["material_identification", "physical_consistency",
                     "terminology", "observation_detail", "reasoning_quality"]
        for field in required:
            if field not in scores or not isinstance(scores[field], (int, float)):
                return {"error": f"Missing or invalid field: {field}"}
            scores[field] = int(scores[field])
        return scores
    except json.JSONDecodeError as e:
        return {"error": f"JSON parse error: {e}"}


# ============================================================
# Aggregation & Reporting
# ============================================================

def aggregate_results(results):
    """Compute summary statistics across all evaluated samples."""
    dimensions = ["material_identification", "physical_consistency",
                   "terminology", "observation_detail", "reasoning_quality"]

    valid = [r for r in results if "error" not in r]
    if not valid:
        return {"error": "No valid results to aggregate"}

    summary = {"num_samples": len(valid), "num_errors": len(results) - len(valid)}

    for dim in dimensions:
        scores = [r[dim] for r in valid]
        summary[dim] = {
            "mean": round(statistics.mean(scores), 2),
            "std": round(statistics.stdev(scores), 2) if len(scores) > 1 else 0.0,
            "min": min(scores),
            "max": max(scores),
        }

    # Overall average
    all_means = [summary[d]["mean"] for d in dimensions]
    summary["overall_mean"] = round(statistics.mean(all_means), 2)

    return summary


def print_report(summary):
    """Print a formatted evaluation report."""
    print("\n" + "=" * 60)
    print("LLM-AS-JUDGE EVALUATION REPORT")
    print("=" * 60)
    print(f"Samples evaluated: {summary['num_samples']}")
    print(f"Evaluation errors: {summary['num_errors']}")
    print("-" * 60)

    dim_names = {
        "material_identification": "Material Identification",
        "physical_consistency": "Physical Consistency",
        "terminology": "Terminology & Professionalism",
        "observation_detail": "Observation Detail",
        "reasoning_quality": "Reasoning Quality",
    }

    for dim, name in dim_names.items():
        s = summary[dim]
        bar = "█" * int(s["mean"]) + "░" * (5 - int(s["mean"]))
        print(f"  {name:<32} {bar}  {s['mean']:.2f} ± {s['std']:.2f}  (range: {s['min']}-{s['max']})")

    print("-" * 60)
    print(f"  {'Overall Average':<32}        {summary['overall_mean']:.2f} / 5.00")
    print("=" * 60)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="LLM-as-Judge Evaluation for Materials VLM")
    parser.add_argument("--image-dir", type=str, help="Directory with test images")
    parser.add_argument("--model", type=str, default="models/Qwen2_5_VL_Physics_RLHF_Merged",
                        help="Path to VLM model")
    parser.add_argument("--input-file", type=str,
                        help="Pre-generated inference results (JSONL: image_path, model_output)")
    parser.add_argument("--openai-key", type=str, help="OpenAI API key")
    parser.add_argument("--anthropic-key", type=str, help="Anthropic API key")
    parser.add_argument("--judge-model", type=str, default="openai",
                        choices=["openai", "claude"], help="Which LLM to use as judge")
    parser.add_argument("--num-samples", type=int, default=30,
                        help="Number of images to evaluate")
    parser.add_argument("--with-image", action="store_true",
                        help="Send image to judge for visual grounding (costs more tokens)")
    parser.add_argument("--output", type=str, default="eval_report.json",
                        help="Output JSON report path")
    args = parser.parse_args()

    # Validate API key
    if args.judge_model == "openai" and not args.openai_key:
        parser.error("--openai-key required when using OpenAI judge")
    if args.judge_model == "claude" and not args.anthropic_key:
        parser.error("--anthropic-key required when using Claude judge")

    # Step 1: Get model outputs
    samples = []

    if args.input_file:
        # Load pre-generated results
        print(f"[Eval] Loading pre-generated outputs from {args.input_file}")
        with open(args.input_file, "r") as f:
            for line in f:
                samples.append(json.loads(line.strip()))
        samples = samples[:args.num_samples]
    elif args.image_dir:
        # Run VLM inference
        image_dir = Path(args.image_dir)
        images = sorted(image_dir.glob("*.jpg")) + sorted(image_dir.glob("*.png"))
        images = images[:args.num_samples]

        if not images:
            print(f"[Error] No images found in {image_dir}")
            return

        model, processor = load_vlm(args.model)

        for i, img_path in enumerate(images):
            print(f"[VLM] Inference {i+1}/{len(images)}: {img_path.name}")
            output = vlm_inference(model, processor, str(img_path))
            samples.append({"image_path": str(img_path), "model_output": output})
            print(f"  Output: {output[:100]}...")

        # Save intermediate results
        intermediate_path = "inference_results.jsonl"
        with open(intermediate_path, "w") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        print(f"[Eval] Inference results saved to {intermediate_path}")
    else:
        parser.error("Either --image-dir or --input-file is required")

    # Step 2: Judge each sample
    print(f"\n[Judge] Evaluating {len(samples)} samples with {args.judge_model}...")
    results = []

    for i, sample in enumerate(samples):
        print(f"[Judge] Scoring {i+1}/{len(samples)}...", end=" ")
        img_path = sample.get("image_path") if args.with_image else None

        if args.judge_model == "openai":
            scores = judge_with_openai(args.openai_key, sample["model_output"], img_path)
        else:
            scores = judge_with_anthropic(args.anthropic_key, sample["model_output"], img_path)

        if "error" in scores:
            print(f"Error: {scores['error'][:80]}")
        else:
            avg = statistics.mean([scores[k] for k in [
                "material_identification", "physical_consistency",
                "terminology", "observation_detail", "reasoning_quality"
            ]])
            print(f"Avg: {avg:.1f}/5 | {scores.get('justification', '')[:60]}")

        scores["image_path"] = sample.get("image_path", "N/A")
        results.append(scores)

        # Rate limiting
        time.sleep(1)

    # Step 3: Aggregate and report
    summary = aggregate_results(results)
    print_report(summary)

    # Save full report
    report = {
        "summary": summary,
        "detailed_results": results,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n[Eval] Full report saved to {args.output}")


if __name__ == "__main__":
    main()
