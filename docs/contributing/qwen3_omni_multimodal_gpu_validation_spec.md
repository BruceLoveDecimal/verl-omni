# Qwen3-Omni Multimodal GPU Validation Spec

Status: working spec

Tracking issue: [#183](https://github.com/verl-project/verl-omni/issues/183)

Implementation base: [#284](https://github.com/verl-project/verl-omni/pull/284)

## Objective

Validate the platform-neutral multimodal Qwen3-Omni Thinker path from PR #284
on NVIDIA GPUs using the existing FSDP + LoRA recipe. The validation should
provide the end-to-end CUDA evidence needed for the two open input-modality
milestones in issue #183:

1. image input training; and
2. audio-conditioned training.

This is validation work, not a second implementation of PR #284. Core dataset,
audio RoPE, reward, and processor changes should be reused as-is unless the GPU
runs expose a CUDA-specific defect.

## Scope

| Scenario | Dataset | Input and output | Purpose |
| --- | --- | --- | --- |
| Image | MMK12 | image + text -> text | Isolate the vision input path and validate multimodal actor/rollout agreement |
| Audio-visual | AVQA-R1-6K | image + audio + text -> text | Validate audio loading, audio RoPE, joint multimodal rollout, and LoRA training |

The following are out of scope:

- Talker or Code2Wav training;
- audio generation;
- video input;
- NPU full-parameter training;
- fully asynchronous full-model weight synchronization; and
- multi-node validation.

An audio-only dataset such as MMAU may be added later if reviewers require
isolated audio evidence. It is not required for the initial validation because
AVQA exercises the audio-conditioned path added by PR #284.

## Base Revision and Reproducibility

Run experiments from an exact PR #284 head SHA and record it in every result.
At the time this spec was written (2026-07-28), the GitHub head was
`98a97c13bfe96fcbc27f48118bc2130d6b9218e9`. The local
`origin/pr/284` ref was still at `5389293`, so it must not be used without
refreshing it.

One safe setup is:

```bash
git fetch origin pull/284/head
git switch -c feat/qwen3-omni-gpu-validation FETCH_HEAD
git rev-parse HEAD
```

PR #284 is still a draft and may be rebased. Re-resolve and record its head
immediately before the final runs. The final submission should be a separate PR
and should contain only the GPU validation recipe, tests, documentation, and
results needed on top of PR #284.

## Target Environment

Baseline hardware:

- one node;
- 4 x NVIDIA H100/H200 80 GB;
- actor and rollout colocated; and
- rollout tensor parallel size 4.

Install from the repository's current pins rather than the older 0.22 versions
still mentioned in the GSPO README:

```bash
uv venv --python 3.12 --seed
source .venv/bin/activate

uv pip install -e ".[gpu]" --torch-backend=auto
uv pip install \
  "vllm-omni @ git+https://github.com/vllm-project/vllm-omni.git@$(cat .github/vllm_omni_pin.txt)"
uv pip install -e ".[train,dev,audio]"
```

The current repository baseline uses `vllm==0.24.0`, the vLLM-Omni commit in
`.github/vllm_omni_pin.txt`, the verl commit in `.github/verl_pin.txt`,
transformers 5.x, and `qwen-omni-utils==0.0.9`.

Before training, capture:

```bash
nvidia-smi
ffmpeg -version
python -c "import torch, transformers, accelerate, vllm; print(torch.__version__, torch.version.cuda, transformers.__version__, accelerate.__version__, vllm.__version__)"
python -c "import verl, verl_omni, vllm_omni, qwen_omni_utils; print('imports OK')"
uv pip freeze
```

Store the command output with the experiment results.

## Validation Plan

### Phase 0: CPU and Wiring Tests

Run the tests introduced or exercised by PR #284:

```bash
pytest -q \
  tests/pipelines/test_qwen3_omni_thinker_adapter_on_cpu.py \
  tests/utils/test_avqa_data_process_on_cpu.py \
  tests/utils/test_omni_rl_datasets_on_cpu.py \
  tests/utils/reward_score/test_choice_reward_on_cpu.py
```

Confirm that:

- AVQA conversion preserves image and audio references;
- `OmniRLHFDataset` returns media in verl's expected image/video/audio order;
- `feature_attention_mask` produces the audio sequence lengths consumed by
  Qwen3-Omni RoPE; and
- the choice reward parses the expected answer tag.

### Phase 1: Existing GPU Regression Smoke

Run the existing tiny-random Qwen3-Omni Thinker smoke before downloading or
loading the full model:

```bash
NUM_GPUS=2 TOTAL_TRAIN_STEPS=2 \
  bash tests/special_e2e/run_gspo_qwen3_omni_thinker_lora_smoke.sh
```

This is a text-only regression gate. It verifies the CUDA FSDP LoRA, rollout,
back-propagation, and weight-sync baseline before multimodal variables are
introduced.

#### Executed result: A800 TP=2 smoke (2026-07-29)

The Phase 1 smoke completed successfully on the PR #284 head
`98a97c13bfe96fcbc27f48118bc2130d6b9218e9`.

Environment:

- hardware: 2 x NVIDIA A800-SXM4-80GB;
- driver: 580.126.09;
- PyTorch: 2.11.0+cu130;
- vLLM: 0.24.0;
- vLLM-Omni: `0.24.1.dev26+gfe478a95a`;
- Transformers: 5.5.3;
- Accelerate: 1.14.0;
- Python: 3.12.3;
- rollout tensor parallel size: 2; and
- total training steps: 2.

The successful run used:

```bash
source /root/autodl-tmp/venvs/verl-omni-pr284/bin/activate
cd /root/autodl-tmp/verl-omni-pr284

PYTHONPATH=/root/autodl-tmp/flash-attn-padding-only \
HF_HOME=/root/autodl-tmp/hf-cache \
HF_ENDPOINT=https://hf-mirror.com \
NUM_GPUS=2 \
TOTAL_TRAIN_STEPS=2 \
MODEL_PATH=/root/autodl-tmp/models/tiny-random/Qwen3-Omni-smoke-v2 \
DATA_DIR=/root/autodl-tmp/smoke-data/math \
bash tests/special_e2e/run_gspo_qwen3_omni_thinker_lora_smoke.sh
```

The process exited with status 0 and printed:

```text
Qwen3-Omni Thinker GSPO+LoRA e2e smoke test passed (training completed successfully).
```

Key metrics:

| Metric | Step 1 | Step 2 | Gate |
| --- | ---: | ---: | --- |
| `training/rollout_actor_probs_pearson_corr` | 0.999789 | 0.999880 | >= 0.95 |
| `rollout_corr/log_ppl_diff` | 0.0000906 | 0.0000996 | abs <= 0.1 |
| `training/rollout_probs_diff_max` | 0.000322 | 0.000306 | finite |
| `actor/entropy` | 5.924358 | 5.924482 | finite |
| `actor/kl_loss` | 5.253e-6 | 3.522e-6 | finite |
| `actor/loss` | 5.253e-9 | 3.522e-9 | finite |
| `actor/grad_norm` | 3.940e-7 | 2.868e-7 | finite |
| `actor/perf/max_memory_allocated_gb` | 2.001 | 2.017 | no OOM |
| `actor/perf/max_memory_reserved_gb` | 2.002 | 2.023 | no OOM |
| `timing_s/update_weights` | 0.789 | 0.816 | completed |
| `timing_s/step` | 25.468 | 8.121 | informational |

This run exercised CUDA FSDP LoRA initialization, TP=2 vLLM-Omni AR rollout,
old and reference log-probability calculation, actor back-propagation, LoRA
weight synchronization, and validation. There was no OOM, NCCL failure, or
CUDA crash. Both GPUs returned to zero allocated process memory after exit.

The complete successful log is stored on the validation machine at:

```text
/root/autodl-tmp/verl-omni-pr284-smoke-logs/tiny-random-2gpu-2step-final.log
```

Environment and script issues found while reaching the successful run:

1. **The offline tiny-model builder is inconsistent with the current model
   registration.** `build_qwen3_omni_tiny_random.py` constructs the model with
   `AutoModelForCausalLM.from_config`, while PR #284 now routes
   `Qwen3OmniMoeForConditionalGeneration` through
   `AutoModelForMultimodalLM`. The offline fallback therefore raises
   `ValueError: Unrecognized configuration class Qwen3OmniMoeConfig`.
2. **The community tiny checkpoint lacks `chat_template.json`.** The rollout
   adapter treats this file as mandatory. The successful run used a
   dereferenced copy of `ShowMaker27/Qwen3-Omni-tiny-random` plus the
   `chat_template.json` from the cached official
   `Qwen3-Omni-30B-A3B-Instruct` checkpoint. The original Hub snapshot and
   official model cache were not modified.
3. **The GPU environment lacks an importable `flash_attn` package.** verl's
   `left_right_2_no_padding` path imports `flash_attn.bert_padding` even though
   the smoke explicitly selects SDPA for model attention. Setting
   `actor_rollout_ref.model.use_remove_padding=False` does not bypass this
   trainer-side conversion.
4. **Full FlashAttention cannot be compiled in the current image.** The system
   CUDA compiler is 12.8, while PyTorch was built for CUDA 13.0. FlashAttention
   2.8.3 correctly rejects this mismatch, and no compatible prebuilt
   Python-3.12/CUDA-13/PyTorch-2.11 wheel was available.
5. **Temporary padding-only workaround.** For this text smoke only, the
   official FlashAttention 2.8.3 `bert_padding.py` and distribution metadata
   were extracted to `/root/autodl-tmp/flash-attn-padding-only`. Its
   `unpad_input` -> `pad_input` round trip was verified on an A800 before the
   run. No FlashAttention CUDA kernel was provided or called; Transformers
   reported that it fell back to native PyTorch attention, consistent with the
   explicit SDPA configuration.
6. **Teardown warning.** After both steps, final validation, and the success
   marker, a DataLoader worker was killed during iterator destruction. The
   exception was ignored, the process exit status remained 0, and no Ray,
   vLLM, or GPU process remained. It should still be monitored in longer runs.

The smoke therefore passes as a CUDA plumbing gate, but it is not an
out-of-the-box pass of the repository script in this image. Before treating
Phase 1 as reproducible CI evidence, fix the tiny builder/checkpoint metadata
and provide a supported FlashAttention dependency or remove the trainer's
unconditional dependency on `flash_attn.bert_padding`.

#### Executed result: H20 TP=2 smoke (2026-07-29)

The Phase 1 smoke was repeated successfully on a 4 x NVIDIA H20 node, using
two GPUs as required by the fixed TP=2 smoke stage config. The run used the
same PR #284 head, `98a97c13bfe96fcbc27f48118bc2130d6b9218e9`.

Environment:

- hardware: 4 x NVIDIA H20 96 GB (2 GPUs used by the smoke);
- driver: 595.71.05;
- PyTorch: 2.11.0+cu130;
- vLLM: 0.24.0;
- vLLM-Omni: `0.24.1.dev26+gfe478a95a`;
- Transformers: 5.5.3;
- Accelerate: 1.14.0;
- rollout tensor parallel size: 2; and
- total training steps: 2.

The process exited with status 0 and printed the normal success marker. Key
metrics were:

| Metric | Step 1 | Step 2 | Gate |
| --- | ---: | ---: | --- |
| `training/rollout_actor_probs_pearson_corr` | 0.999922 | 0.999915 | >= 0.95 |
| `rollout_corr/log_ppl_diff` | 0.0000173 | -0.0001267 | abs <= 0.1 |
| `training/rollout_probs_diff_max` | 0.000215 | 0.000284 | finite |
| `actor/entropy` | 5.923620 | 5.923583 | finite |
| `actor/kl_loss` | 6.055e-6 | 3.364e-6 | finite |
| `actor/loss` | 6.055e-9 | 3.364e-9 | finite |
| `actor/grad_norm` | 4.710e-7 | 9.046e-7 | finite |
| `actor/perf/max_memory_allocated_gb` | 2.001 | 2.063 | no OOM |
| `actor/perf/max_memory_reserved_gb` | 2.002 | 2.080 | no OOM |
| `timing_s/update_weights` | 1.208 | 1.198 | completed |
| `timing_s/step` | 25.097 | 7.517 | informational |

The complete log is stored on the H20 validation node at:

```text
/root/autodl-tmp/logs/h20-smoke.log
```

This repeat exposed three additional builder/environment details:

1. After correcting the builder's AutoModel route, Transformers 5.5 requires
   `rope_parameters` containing `rope_theta`; mutating only the older
   `rope_scaling` field raises `KeyError: 'rope_theta'`.
2. Transformers 5.5 saves the composite processor as
   `processor_config.json`, while the current vLLM-Omni path also expects the
   older flattened `preprocessor_config.json`. The offline checkpoint needed a
   compatible flattened file and `chat_template.json`.
3. The smoke launcher pins Accelerate 1.14.0 even though the installed
   vLLM-Omni package metadata pins Accelerate 1.12.0. The run passed, but this
   dependency conflict should be reconciled for a clean reproducible setup.

The same padding-only FlashAttention workaround was used. The post-training
DataLoader worker `Killed` traceback also reproduced after both steps and
final validation; the success marker and process exit status remained zero.

### Phase 2: Dataset Preparation and Preflight

Prepare MMK12:

```bash
python examples/gspo_trainer/data_process/mmk12.py \
  --local_dataset_path /path/to/mmk12 \
  --local_save_dir /data/mmk12
```

Prepare AVQA-R1-6K:

```bash
python examples/gspo_trainer/data_process/avqa.py \
  --input_dir /path/to/AVQA_R1 \
  --output_dir /data/avqa_r1_6k
```

Record input, kept, dropped, and answer-distribution counts for every split.
Before starting the full model, inspect at least one batch from each dataset:

- MMK12 must produce non-empty image inputs;
- AVQA must produce both non-empty image inputs and non-empty audio inputs;
- AVQA audio must decode successfully at 16 kHz;
- every external AVQA media path must exist; and
- both reward functions must produce a non-zero score on a known-correct
  synthetic response.

AVQA stores absolute media paths in parquet. Every Ray worker must see the same
path. The initial target is single-node, but the constraint still applies to
containers and bind mounts.

### Phase 3: Full-Model Two-Step Smokes

Use `Qwen/Qwen3-Omni-30B-A3B-Instruct` and the existing GPU LoRA launcher.

MMK12:

```bash
TRAIN_FILE=/data/mmk12/train.parquet \
VAL_FILE=/data/mmk12/test.parquet \
MODEL_PATH=/models/Qwen3-Omni-30B-A3B-Instruct \
bash examples/gspo_trainer/qwen3_omni/run_qwen3_omni_thinker_gspo_lora.sh \
  data.max_prompt_length=2048 \
  data.max_response_length=1024 \
  data.val_max_samples=32 \
  reward.custom_reward_function.path=verl_omni/utils/reward_score/mmk12_reward.py \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  trainer.val_before_train=true \
  trainer.test_freq=1 \
  trainer.save_freq=-1 \
  trainer.logger=console \
  trainer.total_training_steps=2 \
  trainer.experiment_name=qwen3_omni_mmk12_gpu_smoke
```

AVQA-R1-6K:

```bash
TRAIN_FILE=/data/avqa_r1_6k/train.parquet \
VAL_FILE=/data/avqa_r1_6k/validation.parquet \
MODEL_PATH=/models/Qwen3-Omni-30B-A3B-Instruct \
bash examples/gspo_trainer/qwen3_omni/run_qwen3_omni_thinker_gspo_lora.sh \
  data.max_prompt_length=2048 \
  data.max_response_length=512 \
  data.val_max_samples=32 \
  data.custom_cls.path=pkg://verl_omni.utils.dataset.omni_rl_datasets \
  data.custom_cls.name=OmniRLHFDataset \
  ++data.mm_processor_kwargs.sampling_rate=16000 \
  reward.custom_reward_function.path=verl_omni/utils/reward_score/choice_reward.py \
  reward.custom_reward_function.name=compute_score \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  trainer.val_before_train=true \
  trainer.test_freq=1 \
  trainer.save_freq=-1 \
  trainer.logger=console \
  trainer.total_training_steps=2 \
  trainer.experiment_name=qwen3_omni_avqa_gpu_smoke
```

The GPU launcher keeps LoRA rank 64, freezes the vision tower, excludes the
audio tower and non-Thinker stages from LoRA, and uses TP=4 for the colocated
rollout.

### Phase 4: Stability and Qualification Runs

After both two-step smokes pass:

1. run each scenario for 20 consecutive steps as the stability gate;
2. inspect metrics and resolve correctness or memory issues before spending
   more GPU time; and
3. for final issue-closing evidence, run 60 steps per scenario and validate at
   steps 0, 20, 40, and 60.

The 20-step runs are sufficient for initial plumbing feedback. The 60-step runs
are recommended for the final PR because the existing text-only GPU milestone
also reports a multi-step training curve. Validation accuracy improvement is
useful evidence but is not a hard correctness gate: short LoRA experiments can
be noisy and this work is intended to validate the training path rather than
tune the recipe.

## Acceptance Criteria

| Area | Required result |
| --- | --- |
| Process exit | Zero exit status, with no Ray, NCCL, CUDA, or media-decoding crash |
| Modality use | MMK12 reaches the model with image tensors; AVQA reaches it with both image and audio tensors |
| Actor/rollout agreement | `training/rollout_actor_probs_pearson_corr >= 0.95` after weight sync |
| Log-prob offset | `abs(rollout_corr/log_ppl_diff) <= 0.1` |
| Numerical stability | Finite actor loss, grad norm, entropy, KL, and reward; no NaN/Inf |
| Reward wiring | Rewards are not all zero and a known-correct response receives positive reward |
| Weight update | LoRA weights synchronize successfully on every completed step |
| Trainable scope | Thinker LoRA parameters are trainable; visual/audio towers and Talker/Code2Wav remain frozen or excluded |
| Memory | No OOM; report peak allocated and reserved memory per GPU |
| Evaluation | Report baseline, final, and peak validation accuracy/reward for both datasets |

Any threshold miss is a failed gate until it is understood. Do not hide a
threshold miss by reporting only the final step.

## Required Evidence

Keep the following artifacts for each run:

- exact verl-omni, verl, and vLLM-Omni commit SHAs;
- complete launch command and resolved Hydra config;
- hardware and dependency snapshot;
- stdout/stderr log;
- dataset conversion statistics;
- per-step actor/rollout Pearson correlation and log-PPL difference;
- actor loss, grad norm, entropy, KL, peak memory, and step time;
- reward and validation accuracy at every evaluation point; and
- a short result table plus training curves suitable for the PR description.

## Submission Plan

The validation should be submitted separately from PR #284, as requested by its
author. A suitable title is:

```text
[omni, tests, doc] test: validate Qwen3-Omni image and audio RL on GPU
```

The patch should avoid duplicating PR #284. Expected deliverables are:

- reproducible GPU commands or a small GPU validation launcher;
- any multimodal GPU smoke assertion that is practical for CI;
- documentation of the exact validated environment and stage configuration;
- result tables and curves; and
- CUDA-specific fixes only if the experiments prove they are needed.

Open PR #290 is not duplicate work: it covers text-only, full-model,
fully-asynchronous rollout and explicitly excludes multimodal end-to-end
support.

Before submission, repeat the repository's duplicate-work checks and include
the results in the PR description. The description must also list every test
command and result, explain why the work does not duplicate an open PR, state
that AI assistance was used, and confirm that the human submitter reviewed
every changed line and can defend the change end to end.

## Known Risks

1. **Moving base:** PR #284 is a draft and can be rebased while experiments are
   running. Always pin and report the exact head.
2. **Version drift:** the GSPO README still describes vLLM/vLLM-Omni 0.22,
   while the current repository pins 0.24-era dependencies. Use the repository
   pins and capture the resolved environment.
3. **Stage-config precedence:** when `stage_configs_path` is set, vLLM-Omni
   takes engine memory and batching values from
   `qwen3_omni_thinker_only.yaml`. A top-level Hydra override such as
   `actor_rollout_ref.rollout.gpu_memory_utilization=0.8` does not change the
   stage value. Start with the GPU stage's `gpu_memory_utilization: 0.4`; if
   tuning is necessary, use an explicit GPU stage-config variant.
4. **Media paths:** AVQA parquet files refer to external absolute paths and are
   not portable without matching mounts.
5. **Filtering:** multimodal placeholder expansion can filter a large fraction
   of AVQA at low prompt-length limits. Report post-filter train and validation
   counts rather than only raw dataset size.
