set -x
export VLLM_USE_V1=0
wandb_token=08c0f4579c7d16938b6090064549af2468fcdeb3
conda activate openrlhf
export PYTHONPATH=/mnt/zheda/default/zly/agentic-o1:$PYTHONPATH
# export RAY_ADDRESS=auto
timestamp=$(date +"%Y%m%d_%H%M%S")
save_path="/mnt/zheda/default/rl_outputs/agentic_test_${timestamp}"
ckpt_path="/mnt/zheda/default/rl_outputs/agentic_test_${timestamp}"

python3 -m openrlhf.cli.train_ppo_ray \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 0 \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 8 \
   --vllm_num_engines 1 \
   --vllm_tensor_parallel_size 8 \
   --init_kl_coef 0.0 \
   --vllm_gpu_memory_utilization 0.9 \
   --gamma 1.0 \
   --eps_clip_low_high 0.2 0.3 \
   --kl_estimator k1 \
   --advantage_estimator group_norm \
   --pretrain /mnt/zheda/default/models/sft_output/global_step60_hf \
   --agent_func_path /mnt/zheda/default/zly/agentic-o1/rl_train/agent_func.py \
   --save_path $save_path \
   --ckpt_path $ckpt_path \
   --save_hf_ckpt \
   --save_steps 30 \
   --micro_train_batch_size 8 \
   --train_batch_size 2048 \
   --micro_rollout_batch_size 256 \
   --rollout_batch_size 256 \
   --n_samples_per_prompt 8 \
   --max_epochs 2 \
   --prompt_max_len 4096 \
   --max_samples 1000000 \
   --generate_max_len 8192 \
   --zero_stage 3 \
   --bf16 \
   --actor_learning_rate 2e-6  \
   --critic_learning_rate 9e-6 \
   --prompt_data /mnt/zheda/default/zly/rl_0724_v5_sampled_processed.jsonl \
   --input_key msg \
   --label_key label \
   --apply_chat_template \
   --gradient_checkpointing \
   --grad_accum_dtype bf16 \
   --packing_samples \
   --vllm_sync_backend nccl \
   --enforce_eager \
   --vllm_enable_sleep \
   --use_liger_kernel \
   --adam_offload \
   --deepspeed_enable_sleep \
   --temperature 1.0 \
   --flash_attn \
   --colocate_all_models \
   --use_dynamic_batch \
   # --use_wandb $wandb_token \

## Clip High (DAPO): Increasing the upper bound of GRPO/PPO’s surrogate loss encourages exploration and stabilizes entropy.
## No KL Loss (DAPO): Eliminating KL loss prevents the LLM from being constrained to the trust region of the original SFT model.
## No Reward Standard Deviation (Dr.GRPO): Removing reward standard deviation removes difficulty bias in GRPO’s loss, ensuring hard and easy problems are better differentiated.
## Length Normalization (Dr.GRPO): Dividing surrogate loss by max context length removes length bias present in GRPO, which increases the length of incorrect responses. replaced by response reward
## Leave One Out (Loop/RLOO): Removing one sample for advantage estimation reduces variance for policy gradient without introducing bias.
## Compact Filtering (Us): Inspired by DAPO, we mask the loss for trajectories that reach max context length, timeout during generation (20 minutes), or reach maximum steps. Described further below.
## No Entropy Loss (Us): Entropy loss introduces higher instability and eventually leads to exponentially increasing entropy, which collapses training. Provided that the base model’s token-level entropy is within 0.3-1, entropy loss is not needed.

# USE K1 estimator

# Lower learning rate for actor to avoid divergence