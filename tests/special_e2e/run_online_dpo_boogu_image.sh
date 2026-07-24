#!/usr/bin/env bash
# Boogu-Image online DPO e2e smoke test (minimal runtime), vllm_omni rollout.
#
# Flow: dummy parquet -> vllm_omni rollout (BooguImageDPOPipeline, ODE) ->
#       jpeg_compressibility reward -> online DPO pairing -> ref noise pred ->
#       FSDP LoRA actor update via DPODiffusersFSDPEngine.
#
# Requires: vllm-omni (>= Boogu support), the `boogu-image` package,
#   tiny Boogu-Image at ~/models/tiny-random/Boogu-Image
#   (see tests/special_e2e/build_boogu_image_tiny_random.py).
set -euo pipefail

NUM_GPUS=${NUM_GPUS:-2}
MODEL_PATH=${MODEL_PATH:-${HOME}/models/tiny-random/Boogu-Image}
TOKENIZER_PATH=${TOKENIZER_PATH:-${MODEL_PATH}/processor}
DATA_DIR=${DATA_DIR:-${HOME}/data/dummy_diffusion}
TRAIN_FILES=${TRAIN_FILES:-${DATA_DIR}/train.parquet}
VAL_FILES=${VAL_FILES:-${DATA_DIR}/test.parquet}
TOTAL_TRAIN_STEPS=${TOTAL_TRAIN_STEPS:-1}

ENGINE=vllm_omni
max_prompt_length=256

if ! python3 -c 'import boogu' >/dev/null 2>&1; then
    echo "FAIL: the boogu-image package is required for the training-side transformer."
    exit 1
fi

ATTN_BACKEND=native
ROLLOUT_ATTN_BACKEND=TORCH_SDPA

# Online DPO needs at least two candidates per prompt for win/reject pairing.
n_resp_per_prompt=2
micro_bsz_per_gpu=2
micro_bsz=$((micro_bsz_per_gpu * NUM_GPUS))
mini_bsz=${micro_bsz}
train_batch_size=${mini_bsz}

python3 tests/special_e2e/create_dummy_diffusion_data.py \
    --local_save_dir "${DATA_DIR}" \
    --train_size "${train_batch_size}" \
    --val_size 4

python3 -m verl_omni.trainer.main_diffusion \
    algorithm.trainer_type=direct_preference \
    algorithm.sample_source=online \
    algorithm.paired_preference=true \
    data.train_files=${TRAIN_FILES} \
    data.val_files=${VAL_FILES} \
    data.train_batch_size=${train_batch_size} \
    data.max_prompt_length=${max_prompt_length} \
    actor_rollout_ref.model.algorithm=dpo \
    actor_rollout_ref.model.model_type=diffusion_dpo_model \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.model.tokenizer_path=${TOKENIZER_PATH} \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.model.attn_backend=${ATTN_BACKEND} \
    actor_rollout_ref.rollout.rollout_attn_backend=${ROLLOUT_ATTN_BACKEND} \
    actor_rollout_ref.model.lora_rank=8 \
    actor_rollout_ref.model.lora_alpha=16 \
    actor_rollout_ref.model.target_modules=all-linear \
    actor_rollout_ref.model.fsdp_layer_prefixes="['double_stream_layers.','single_stream_layers.','context_refiner.','noise_refiner.','ref_image_refiner.']" \
    actor_rollout_ref.actor.diffusion_loss.loss_mode=dpo \
    actor_rollout_ref.actor.optim.lr=1e-5 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${mini_bsz} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${micro_bsz_per_gpu} \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=${ENGINE} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.calculate_log_probs=false \
    actor_rollout_ref.rollout.agent.num_workers=1 \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.rollout.pipeline.num_inference_steps=4 \
    actor_rollout_ref.rollout.pipeline.height=256 \
    actor_rollout_ref.rollout.pipeline.width=256 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.pipeline.guidance_scale=4.0 \
    actor_rollout_ref.rollout.pipeline.max_sequence_length=${max_prompt_length} \
    actor_rollout_ref.rollout.val_kwargs.pipeline.num_inference_steps=4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${micro_bsz_per_gpu} \
    reward.num_workers=1 \
    reward.reward_model.enable=False \
    trainer.logger=console \
    trainer.project_name=verl-test \
    trainer.experiment_name=online-dpo-boogu-image-e2e \
    trainer.log_val_generations=0 \
    trainer.n_gpus_per_node=${NUM_GPUS} \
    trainer.nnodes=1 \
    trainer.val_before_train=False \
    trainer.test_freq=-1 \
    trainer.save_freq=-1 \
    trainer.resume_mode=disable \
    trainer.total_training_steps=${TOTAL_TRAIN_STEPS} \
    "$@"

echo "Online DPO Boogu-Image e2e test passed (training completed successfully)."
