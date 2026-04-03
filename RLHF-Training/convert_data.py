import pandas as pd

# 读取你的 jsonl 文件
df = pd.read_json("rl_run/data/physics_rl_train.jsonl", lines=True)

# 把它保存为 parquet 文件
output_path = "rl_run/data/physics_rl_train.parquet"
df.to_parquet(output_path, engine='pyarrow')
print(f"数据已成功转换为: {output_path}")
