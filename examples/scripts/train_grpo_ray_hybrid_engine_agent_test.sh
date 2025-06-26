set -x
export VLLM_USE_V1=0
wandb_token=08c0f4579c7d16938b6090064549af2468fcdeb3
export PYTHONPATH=/home/zjuici/zly/agentic-o1:$PYTHONPATH

python3 -m openrlhf.cli.train_ppo_ray \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 1 \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 4 \
   --vllm_num_engines 2 \
   --vllm_tensor_parallel_size 1 \
   --vllm_gpu_memory_utilization 0.95 \
   --init_kl_coef 1e-3 \
   --gamma 1.0 \
   --use_kl_loss \
   --kl_estimator k3 \
   --advantage_estimator group_norm \
   --pretrain /data/model/Qwen2.5-7B-Instruct \
   --agent_func_path /home/zjuici/zly/agentic-o1/rl_train/agent_func.py \
   --save_path /data/sft_outputs/rl_checkpoint/agentic_test \
   --ckpt_path /data/sft_outputs/rl_checkpoint/agentic_test \
   --save_hf_ckpt \
   --save_steps 1000 \
   --micro_train_batch_size 1 \
   --train_batch_size 512 \
   --micro_rollout_batch_size 4 \
   --rollout_batch_size 128 \
   --n_samples_per_prompt 4 \
   --max_epochs 2 \
   --prompt_max_len 4096 \
   --max_samples 1000000 \
   --generate_max_len 8192 \
   --zero_stage 3 \
   --bf16 \
   --actor_learning_rate 1e-6 \
   --critic_learning_rate 9e-6 \
   --prompt_data /home/zjuici/zly/filtered_data_messages.jsonl \
   --input_key instruction \
   --label_key label \
   --apply_chat_template \
   --normalize_reward \
   --gradient_checkpointing \
   --grad_accum_dtype bf16 \
   --packing_samples \
   --vllm_sync_backend nccl \
   --enforce_eager \
   --vllm_enable_sleep \
   --use_liger_kernel \
   --adam_offload \
   --deepspeed_enable_sleep \
   --flash_attn \
   --entropy_loss_coef 0.001 \
   --use_wandb $wandb_token

# You could also try
#   --kl_estimator k2 \
# dont use kl_target , keep kl increasing

