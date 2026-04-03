nproc_per_node=1 # 如果你有两张显卡就改2，一张就改1

CUDA_VISIBLE_DEVICES=0,1 swift rlhf \
    --rlhf_type grpo \
    --model_type qwen2_5-vl-7b-instruct \
    --model_id_or_path ./models/qwen2_5_vl_physics_merged \
    --reward_funcs physics_reward_func \
    --dataset ./rl_run/data/physics_rl_train.parquet \
    --learning_rate 1e-6 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --max_length 1024 \
    --num_train_epochs 1 \
    --vllm_enable true \
    --vllm_gpu_memory_utilization 0.4 \
    --output_dir ./rl_run/outputs/swift_grpo
