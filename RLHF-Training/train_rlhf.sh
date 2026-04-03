export VLLM_LENIENT_PADDING=1

CUDA_VISIBLE_DEVICES=0 swift rlhf \
    --rlhf_type grpo \
    --model_type qwen2_5_vl \
    --model models/qwen2_5_vl_physics_merged \
    --reward_funcs material_physics_score \
    --custom_register_path ./swift_physics_reward.py \
    --dataset data/physics_rl_train.parquet \
    --max_new_tokens 512 \
    --num_sample_generations 4 \
    --learning_rate 1e-6 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --max_length 1024 \
    --num_train_epochs 1 \
    --use_vllm false \
     \
     \
    --output_dir outputs/swift_grpo \
    --warmup_steps 10 \
    --bf16 true \
    --logging_steps 1 \
    --template qwen2_5_vl \
    --temperature 0.8
