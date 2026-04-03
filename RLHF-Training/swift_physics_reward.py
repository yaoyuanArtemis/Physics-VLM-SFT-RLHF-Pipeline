from swift.rewards import orms

class MaterialPhysicsScore:
    def __init__(self, args=None, **kwargs):
        # 框架初始化时会把参数传进来，我们在这里接收
        self.args = args

    def __call__(self, completions, **kwargs) -> list[float]:
        # 框架在训练时，每次会把生成的文本传给 __call__
        ground_truths = kwargs.get('solution', kwargs.get('answer', []))
        
        if isinstance(ground_truths, str):
            ground_truths = [ground_truths] * len(completions)
        elif not ground_truths:
            ground_truths = [""] * len(completions)
        elif len(ground_truths) == 1 and len(completions) > 1:
            ground_truths = [ground_truths[0]] * len(completions)
            
        rewards = []
        for sol_raw, gt_raw in zip(completions, ground_truths):
            score = 0.0
            sol = str(sol_raw).lower()
            gt = str(gt_raw).lower()

            # --- 1. 核心视觉特征 ---
            vision_keywords = [
                "grain boundary", "晶界", "interface", "界面", "dislocation", "位错", 
                "twin", "孪晶", "precipitate", "析出相", "lamellar", "层片",
                "martensite", "马氏体", "void", "微孔", "segregation", "偏聚",
                "coherent", "共格", "semicoherent", "半共格", "misfit", "错配"
            ]
            for kw in vision_keywords:
                if kw in gt and kw in sol:
                    score += 0.5

            # --- 2. 计算属性关联 ---
            physics_keywords = [
                "formation energy", "形成能", "stacking fault", "层错", "sfe", 
                "elastic", "弹性", "thermodynamic", "热力学", "kinetic", "动力学",
                "deep potential", "dp", "molecular dynamics", "分子动力学", "md",
                "dft", "第一性原理", "phase field", "相场", "energy barrier", "能垒"
            ]
            for kw in physics_keywords:
                if kw in gt and kw in sol:
                    score += 0.8

            # --- 3. 严惩尺度混乱 ---
            scale_exclusive = [
                (["atomistic", "原子级", "vacancy", "空位", "interstitial"], ["macroscopic", "宏观", "cm", "mm"]),
                (["tem", "hrtem", "高分辨", "lattice"], ["om", "金相显微镜", "低倍"])
            ]
            for true_scale, false_scale in scale_exclusive:
                if any(t in gt for t in true_scale) and any(f in sol for f in false_scale):
                    score -= 2.0 

            # --- 4. 严惩晶体学相变错误 ---
            if ("bcc" in gt or "beta" in gt) and ("hcp" in gt or "alpha" in gt):
                if ("hcp" in sol or "alpha" in sol) and not ("bcc" in sol or "beta" in sol):
                    score -= 2.5

            # --- 5. 封杀“灌水”词汇 ---
            bs_keywords = ["artistic", "beautiful", "艺术", "漂亮", "aesthetic", "美观"]
            if any(bs in sol for bs in bs_keywords):
                score -= 1.5

            # --- 6. 最终精确匹配 ---
            if gt in sol:
                score += 3.0
            length_bonus = min(len(sol) / 1000.0, 0.5) 
            score += length_bonus
            rewards.append(float(max(score, -5.0)))
        
        return rewards

# 直接将这个类注入到 SWIFT 的字典中
orms["material_physics_score"] = MaterialPhysicsScore
print("✅ [Plugin] 物理材料打分器 (material_physics_score) 类注入成功！")