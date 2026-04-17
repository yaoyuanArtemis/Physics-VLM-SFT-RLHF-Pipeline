"""
Material Science Agent Demo
============================
A lightweight multi-agent demonstration that integrates the fine-tuned
VLM (visual perception) with external materials science tools (action).

Architecture:
    User Image → VLM Tool (perception) → Materials Project API (knowledge)
              → Agent LLM (reasoning & synthesis) → Comprehensive Report

Requirements:
    pip install transformers torch qwen-vl-utils mp-api langchain-core

Usage:
    python agent_demo.py --image assets/mat_img_686.jpg
    python agent_demo.py --image assets/mat_img_686.jpg --mp-api-key YOUR_KEY
"""

import argparse
import json
import re
import torch
from pathlib import Path

# ============================================================
# Tool 1: Material-Aware VLM (Visual Perception)
# ============================================================

class VLMTool:
    """Wraps the fine-tuned Qwen2.5-VL-7B-Physics-RLHF model as a callable tool."""

    def __init__(self, model_path: str = "models/Qwen2_5_VL_Physics_RLHF_Merged"):
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        print("[VLM Tool] Loading model...")
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        print("[VLM Tool] Model loaded.")

    def analyze(self, image_path: str, prompt: str = None) -> str:
        """Analyze a materials microscopy image and return textual description."""
        from qwen_vl_utils import process_vision_info

        if prompt is None:
            prompt = "Please describe this material microstructure in detail."

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"file://{Path(image_path).resolve()}"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt"
        ).to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, max_new_tokens=512, temperature=0.7)

        generated = output_ids[0][inputs.input_ids.shape[1]:]
        return self.processor.decode(generated, skip_special_tokens=True)


# ============================================================
# Tool 2: Materials Project API (Knowledge Retrieval)
# ============================================================

class MaterialsProjectTool:
    """Queries the Materials Project database for material properties via REST API."""

    MP_API_URL = "https://api.materialsproject.org/materials/summary/"

    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self.available = False
        if api_key:
            try:
                import requests
                resp = requests.get(
                    self.MP_API_URL,
                    params={"formula": "Si", "_fields": "material_id", "_limit": "1"},
                    headers={"X-API-KEY": api_key},
                    timeout=10,
                )
                if resp.status_code == 200:
                    self.available = True
                    print("[MP Tool] Connected to Materials Project (REST API).")
                else:
                    print(f"[MP Tool] API returned status {resp.status_code}: {resp.text[:200]}")
                    print("[MP Tool] Falling back to mock data.")
            except Exception as e:
                print(f"[MP Tool] Connection failed: {e}, using mock data.")

    def query(self, formula: str) -> dict:
        """Query material properties by chemical formula."""
        if not self.available:
            return self._mock_query(formula)
        try:
            import requests
            resp = requests.get(
                self.MP_API_URL,
                params={
                    "formula": formula,
                    "_fields": "material_id,formula_pretty,symmetry,band_gap,"
                               "formation_energy_per_atom,bulk_modulus,shear_modulus",
                    "_limit": "1",
                },
                headers={"X-API-KEY": self.api_key},
                timeout=15,
            )
            if resp.status_code != 200:
                return {"error": f"API status {resp.status_code}: {resp.text[:200]}"}
            data = resp.json().get("data", [])
            if data:
                d = data[0]
                sym = d.get("symmetry") or {}
                bulk = d.get("bulk_modulus") or {}
                shear = d.get("shear_modulus") or {}
                return {
                    "material_id": d.get("material_id", "N/A"),
                    "formula": d.get("formula_pretty", formula),
                    "crystal_system": sym.get("crystal_system", "N/A"),
                    "space_group": sym.get("symbol", "N/A"),
                    "band_gap_eV": d.get("band_gap"),
                    "formation_energy_eV_atom": round(d["formation_energy_per_atom"], 4) if d.get("formation_energy_per_atom") is not None else None,
                    "bulk_modulus_GPa": round(bulk["vrh"], 1) if bulk.get("vrh") else None,
                    "shear_modulus_GPa": round(shear["vrh"], 1) if shear.get("vrh") else None,
                }
            return {"error": f"No data found for {formula}"}
        except Exception as e:
            return {"error": str(e)}

    def _mock_query(self, formula: str) -> dict:
        """Return mock data when API key is not available (for demo purposes)."""
        mock_db = {
            "Fe": {"material_id": "mp-13", "formula": "Fe", "crystal_system": "cubic",
                   "space_group": "Im-3m", "band_gap_eV": 0.0,
                   "formation_energy_eV_atom": 0.0, "bulk_modulus_GPa": 170.3,
                   "shear_modulus_GPa": 81.6},
            "Cu": {"material_id": "mp-30", "formula": "Cu", "crystal_system": "cubic",
                   "space_group": "Fm-3m", "band_gap_eV": 0.0,
                   "formation_energy_eV_atom": 0.0, "bulk_modulus_GPa": 137.0,
                   "shear_modulus_GPa": 46.8},
            "TiO2": {"material_id": "mp-2657", "formula": "TiO2", "crystal_system": "tetragonal",
                     "space_group": "P4_2/mnm", "band_gap_eV": 1.78,
                     "formation_energy_eV_atom": -3.21, "bulk_modulus_GPa": 211.0,
                     "shear_modulus_GPa": 89.2},
            "Al2O3": {"material_id": "mp-1143", "formula": "Al2O3", "crystal_system": "trigonal",
                      "space_group": "R-3c", "band_gap_eV": 5.85,
                      "formation_energy_eV_atom": -3.48, "bulk_modulus_GPa": 228.9,
                      "shear_modulus_GPa": 147.3},
        }
        if formula in mock_db:
            return mock_db[formula]
        return {"note": f"Mock mode: no data for {formula}. Connect MP API for real queries."}


# ============================================================
# Tool 3: Physics Calculator
# ============================================================

class PhysicsCalculatorTool:
    """Performs common materials science calculations."""

    @staticmethod
    def hall_petch(sigma_0: float, k: float, d: float) -> dict:
        """Hall-Petch equation: σ_y = σ_0 + k / √d"""
        sigma_y = sigma_0 + k / (d ** 0.5)
        return {
            "equation": "σ_y = σ_0 + k / √d",
            "sigma_0_MPa": sigma_0, "k_MPa_um": k, "grain_size_um": d,
            "yield_strength_MPa": round(sigma_y, 2),
        }

    @staticmethod
    def bragg(d_spacing_nm: float, wavelength_nm: float = 0.0251) -> dict:
        """Bragg's law: 2d sinθ = nλ, solve for θ (n=1)."""
        import math
        sin_theta = wavelength_nm / (2 * d_spacing_nm)
        if abs(sin_theta) > 1:
            return {"error": "No valid diffraction angle for given d-spacing."}
        theta_rad = math.asin(sin_theta)
        return {
            "equation": "2d sinθ = nλ",
            "d_spacing_nm": d_spacing_nm, "wavelength_nm": wavelength_nm,
            "bragg_angle_deg": round(math.degrees(theta_rad), 3),
            "two_theta_deg": round(2 * math.degrees(theta_rad), 3),
        }


# ============================================================
# Agent: Orchestrator
# ============================================================

class MaterialScienceAgent:
    """
    A simple rule-based agent that orchestrates the three tools.

    Workflow:
    1. VLM analyzes the input image (perception)
    2. Extract material keywords from VLM output
    3. Query Materials Project for relevant properties (knowledge)
    4. Optionally perform physics calculations (reasoning)
    5. Synthesize a comprehensive report
    """

    MATERIAL_PATTERNS = [
        (r'\b(Fe|iron)\b', 'Fe'), (r'\b(Cu|copper)\b', 'Cu'),
        (r'\b(Al2O3|alumina|corundum)\b', 'Al2O3'), (r'\b(TiO2|titania|rutile)\b', 'TiO2'),
        (r'\b(Ni|nickel)\b', 'Ni'), (r'\b(Ti|titanium)\b', 'Ti'),
        (r'\b(Au|gold)\b', 'Au'), (r'\b(Ag|silver)\b', 'Ag'),
        (r'\b(SiC|silicon carbide)\b', 'SiC'), (r'\b(ZnO|zinc oxide)\b', 'ZnO'),
        (r'\b(PbS|lead sulfide)\b', 'PbS'), (r'\b(Pd|palladium)\b', 'Pd'),
        (r'\bFCC\b', None), (r'\bBCC\b', None), (r'\bHCP\b', None),
    ]

    def __init__(self, vlm_tool, mp_tool, calc_tool):
        self.vlm = vlm_tool
        self.mp = mp_tool
        self.calc = calc_tool

    def extract_materials(self, text: str) -> list:
        """Extract material formulas from VLM output."""
        found = []
        for pattern, formula in self.MATERIAL_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE) and formula:
                if formula not in found:
                    found.append(formula)
        return found

    def extract_grain_size(self, text: str) -> float | None:
        """Try to extract grain size from VLM output."""
        match = re.search(r'(\d+[\.\d]*)\s*(?:nm|nanometer)', text, re.IGNORECASE)
        if match:
            return float(match.group(1)) / 1000.0  # convert nm to µm
        match = re.search(r'(\d+[\.\d]*)\s*(?:µm|um|micrometer|micron)', text, re.IGNORECASE)
        if match:
            return float(match.group(1))
        return None

    def run(self, image_path: str, user_prompt: str = None) -> dict:
        """Execute the full agent pipeline."""
        report = {"image": image_path, "steps": []}

        # Step 1: Visual Analysis
        print("\n[Agent] Step 1: Running VLM visual analysis...")
        vlm_prompt = user_prompt or "Please describe this material microstructure in detail. Identify the material type, crystal structure, defects, and any notable features."
        vlm_output = self.vlm.analyze(image_path, vlm_prompt)
        report["steps"].append({
            "tool": "VLM (Material-Aware Vision-Language Model)",
            "action": "Analyze microscopy image",
            "output": vlm_output,
        })
        print(f"[Agent] VLM output: {vlm_output[:200]}...")

        # Step 2: Extract materials and query database
        materials = self.extract_materials(vlm_output)
        print(f"[Agent] Step 2: Extracted materials: {materials}")

        for formula in materials[:3]:  # limit to top 3
            print(f"[Agent] Querying Materials Project for {formula}...")
            mp_result = self.mp.query(formula)
            report["steps"].append({
                "tool": "Materials Project API",
                "action": f"Query properties for {formula}",
                "output": mp_result,
            })

        # Step 3: Physics calculations if applicable
        grain_size = self.extract_grain_size(vlm_output)
        if grain_size:
            print(f"[Agent] Step 3: Computing Hall-Petch (grain size={grain_size} µm)...")
            hp = self.calc.hall_petch(sigma_0=70.0, k=0.74, d=grain_size)
            report["steps"].append({
                "tool": "Physics Calculator",
                "action": f"Hall-Petch yield strength estimation (d={grain_size} µm)",
                "output": hp,
            })

        # Step 4: Synthesize report
        report["summary"] = self._synthesize(report["steps"])
        return report

    def _synthesize(self, steps: list) -> str:
        """Combine all tool outputs into a final summary."""
        lines = ["=" * 60, "MATERIAL SCIENCE AGENT — COMPREHENSIVE REPORT", "=" * 60, ""]

        for i, step in enumerate(steps, 1):
            lines.append(f"--- Step {i}: {step['tool']} ---")
            lines.append(f"Action: {step['action']}")
            if isinstance(step['output'], dict):
                for k, v in step['output'].items():
                    lines.append(f"  {k}: {v}")
            else:
                lines.append(str(step['output']))
            lines.append("")

        lines.append("=" * 60)
        return "\n".join(lines)


# ============================================================
# Main Entry Point
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Material Science Agent Demo")
    parser.add_argument("--image", type=str, required=True, help="Path to microscopy image")
    parser.add_argument("--model", type=str, default="models/Qwen2_5_VL_Physics_RLHF_Merged",
                        help="Path to the fine-tuned VLM")
    parser.add_argument("--mp-api-key", type=str, default=None,
                        help="Materials Project API key (optional, uses mock data if not provided)")
    parser.add_argument("--prompt", type=str, default=None, help="Custom analysis prompt")
    parser.add_argument("--output", type=str, default="agent_report.json",
                        help="Output JSON file for the report")
    args = parser.parse_args()

    # Initialize tools
    vlm_tool = VLMTool(args.model)
    mp_tool = MaterialsProjectTool(args.mp_api_key)
    calc_tool = PhysicsCalculatorTool()

    # Initialize and run agent
    agent = MaterialScienceAgent(vlm_tool, mp_tool, calc_tool)
    report = agent.run(args.image, args.prompt)

    # Output
    print("\n" + report["summary"])

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n[Agent] Full report saved to {args.output}")


if __name__ == "__main__":
    main()
