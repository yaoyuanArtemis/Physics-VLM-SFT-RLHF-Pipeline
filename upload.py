from huggingface_hub import HfApi

api = HfApi()

# 填入你喜欢的仓库名
repo_id = "yaoyuanlf/Qwen2.5-VL-7B-Physics-RLHF"

print(f"开始上传至: {repo_id} ...")

# 1. 创建仓库（如果已存在会跳过）
api.create_repo(repo_id=repo_id, exist_ok=True)

# 2. 上传整个合并后的模型文件夹
api.upload_folder(
    folder_path="models/Qwen2_5_VL_Physics_RLHF_Merged",
    repo_id=repo_id,
    repo_type="model"
)

print("🎉 上传成功！去 Hugging Face 看看你的成果吧！")
