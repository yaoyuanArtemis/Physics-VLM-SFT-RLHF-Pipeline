from swift.ui.app import SwiftWebUI
from swift.arguments import AppArguments


real_model_path = '/root/.cache/modelscope/hub/models/Qwen/Qwen2___5-VL-7B-Instruct'
# 1. 在这里直接定义端口和主机地址
args = AppArguments(
    model=real_model_path,
    model_type='qwen2_5_vl',
    adapters=['rl_run/outputs/swift_grpo/v42-20260401-162133/checkpoint-1500'],
    template='qwen2_5_vl',
    infer_backend='transformers',
    stream=True,
    server_name='0.0.0.0', # 允许外部访问
    server_port=6006       # 对应 AutoDL 的自定义服务端口
)

# 2. 这里的 main() 不需要再传参
if __name__ == '__main__':
    SwiftWebUI(args).main()