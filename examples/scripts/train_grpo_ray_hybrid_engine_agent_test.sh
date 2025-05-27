set -x

export VLLM_USE_V1=0
wandb_token=08c0f4579c7d16938b6090064549af2468fcdeb3
export PYTHONPATH=/home/zjuici/zly/agentic-o1:$PYTHONPATH

python3 -m openrlhf.cli.train_ppo_ray \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 1 \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 1 \
   --vllm_num_engines 1 \
   --vllm_tensor_parallel_size 1 \
   --vllm_gpu_memory_utilization 0.6 \
   --init_kl_coef 1e-3 \
   --gamma 1.0 \
   --use_kl_loss \
   --kl_estimator k3 \
   --advantage_estimator group_norm \
   --pretrain /data/model/Qwen3-8B \
   --agent_func_path /home/zjuici/zly/agentic-o1/rl_train/agent_func.py \
   --save_path /data/sft_outputs/rl_checkpoint/agentic_test \
   --ckpt_path /data/sft_outputs/rl_checkpoint/agentic_test \
   --save_hf_ckpt \
   --micro_train_batch_size 2 \
   --train_batch_size 8 \
   --micro_rollout_batch_size 2 \
   --rollout_batch_size 8 \
   --n_samples_per_prompt 2 \
   --max_epochs 1 \
   --prompt_max_len 8192 \
   --max_samples 100000 \
   --generate_max_len 2048 \
   --zero_stage 3 \
   --bf16 \
   --actor_learning_rate 5e-7 \
   --critic_learning_rate 9e-6 \
   --prompt_data /home/zjuici/zly/test_samples.json \
   --input_key prompt \
   --label_key label \
   --apply_chat_template \
   --normalize_reward \
   --gradient_checkpointing \
   --packing_samples \
   --vllm_sync_backend nccl \
   --enforce_eager \
   --vllm_enable_sleep \
   --use_liger_kernel \
   --adam_offload \
   --deepspeed_enable_sleep \
   --flash_attn \
   --use_wandb $wandb_token

# You could also try
#   --kl_estimator k2 \

