# Boogu-Image DiffusionNFT LoRA, vllm_omni rollout
#
# Prerequisites: same as examples/flowgrpo_trainer/boogu_image/README.md
# (boogu-image package + trust_remote_code).
#
# DiffusionNFT: the "old" LoRA adapter generates rollout.n candidates per
# prompt via the deterministic ODE with standard Boogu text CFG; the reward
# model scores them; group-relative advantages become reward_prob weights for
# the forward-process negative-aware objective (policy / old / ref forwards,
# CFG-free at training time). The old adapter refreshes by EMA per
# algorithm.old_policy_* below.
set -x

WORKSPACE=${WORKSPACE:-$HOME}

ocr_train_path=$WORKSPACE/data/ocr/boogu_image/train.parquet
ocr_test_path=$WORKSPACE/data/ocr/boogu_image/test.parquet

model_name=Boogu/Boogu-Image-0.1-Base
reward_model_name=Qwen/Qwen3-VL-8B-Instruct
reward_function_path=verl_omni/utils/reward_score/genrm_ocr.py

NUM_GPUS_ACTOR_ROLLOUT_REWARD=4
# The vllm-omni BooguImagePipeline supports neither TP nor SP nor CFG-parallel.
ROLLOUT_TP=1
REWARD_TP=4

ENGINE=vllm_omni
REWARD_ENGINE=vllm

python3 -m verl_omni.trainer.main_diffusion \
    algorithm.trainer_type=direct_preference \
    algorithm.sample_source=online \
    algorithm.paired_preference=false \
    algorithm.timestep_fraction=0.6 \
    algorithm.old_policy_decay_schedule=delayed_linear_to_0_999 \
    algorithm.old_policy_update_interval=1 \
    algorithm.adv_mode=continuous \
    data.train_files=$ocr_train_path \
    data.val_files=$ocr_test_path \
    data.train_batch_size=16 \
    data.max_prompt_length=256 \
    actor_rollout_ref.model.algorithm=diffusion_nft \
    actor_rollout_ref.model.model_type=diffusion_nft_model \
    actor_rollout_ref.model.path=$model_name \
    actor_rollout_ref.model.tokenizer_path=$model_name/processor \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.model.lora_rank=64 \
    actor_rollout_ref.model.lora_alpha=128 \
    actor_rollout_ref.model.policy_state_adapters='["default","old"]' \
    actor_rollout_ref.model.target_modules="['to_q','to_k','to_v','to_out.0','img_to_q','img_to_k','img_to_v','img_out','instruct_to_q','instruct_to_k','instruct_to_v','instruct_out','feed_forward.linear_1','feed_forward.linear_2','feed_forward.linear_3','img_feed_forward.linear_1','img_feed_forward.linear_2','img_feed_forward.linear_3']" \
    actor_rollout_ref.model.fsdp_layer_prefixes="['double_stream_layers.','single_stream_layers.','context_refiner.','noise_refiner.','ref_image_refiner.']" \
    actor_rollout_ref.actor.diffusion_loss.loss_mode=diffusion_nft \
    actor_rollout_ref.actor.diffusion_loss.mix_beta=0.5 \
    actor_rollout_ref.actor.diffusion_loss.ref_kl_coef=0.001 \
    actor_rollout_ref.actor.diffusion_loss.adv_clip_max=5.0 \
    actor_rollout_ref.actor.optim.lr=1e-4 \
    actor_rollout_ref.actor.optim.weight_decay=0.0001 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.calculate_log_probs=false \
    actor_rollout_ref.rollout.rollout_adapter=old \
    actor_rollout_ref.rollout.agent.num_workers=$((NUM_GPUS_ACTOR_ROLLOUT_REWARD / ROLLOUT_TP)) \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.rollout.pipeline.num_inference_steps=35 \
    actor_rollout_ref.rollout.pipeline.height=1024 \
    actor_rollout_ref.rollout.pipeline.width=1024 \
    actor_rollout_ref.rollout.pipeline.guidance_scale=4.0 \
    actor_rollout_ref.rollout.pipeline.max_sequence_length=256 \
    actor_rollout_ref.rollout.val_kwargs.pipeline.num_inference_steps=50 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    reward.num_workers=$((NUM_GPUS_ACTOR_ROLLOUT_REWARD / REWARD_TP)) \
    reward.reward_model.enable=True \
    reward.reward_model.model_path=$reward_model_name \
    reward.reward_model.rollout.name=$REWARD_ENGINE \
    reward.reward_model.rollout.tensor_model_parallel_size=$REWARD_TP \
    reward.custom_reward_function.path=$reward_function_path \
    reward.custom_reward_function.name=compute_score_ocr \
    trainer.logger='["console", "wandb"]' \
    trainer.project_name=diffusion_nft \
    trainer.experiment_name=boogu_image_diffusionnft_lora \
    trainer.log_val_generations=8 \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=$NUM_GPUS_ACTOR_ROLLOUT_REWARD \
    trainer.nnodes=1 \
    trainer.save_freq=30 \
    trainer.test_freq=30 \
    trainer.total_epochs=15 \
    trainer.total_training_steps=300 "$@"
