# Train Boogu-Image with DiffusionNFT

[DiffusionNFT](https://github.com/NVlabs/DiffusionNFT) (negative-aware
finetuning) optimizes the policy on the **forward diffusion process**: the
"old" LoRA adapter generates candidates online, rewards become group-relative
`reward_prob` weights, and the loss mixes an explicit positive and an
implicit negative prediction (`mix_beta`) over forward-noised clean latents —
no reverse-trajectory log-probs.

## Prerequisites

Same as [FlowGRPO for Boogu-Image](../../flowgrpo_trainer/boogu_image/README.md):

- vllm-omni at (or past) the repo pin, with Boogu-Image support;
- the canonical `boogu-image` package for the training-side transformer:

```bash
pip install "boogu-image @ git+https://github.com/boogu-project/Boogu-Image.git"
```

- `actor_rollout_ref.model.trust_remote_code=true` (already set in the
  launch script);
- OCR train/val parquet built by
  [`examples/flowgrpo_trainer/data_process/boogu_image_ocr.py`](../../flowgrpo_trainer/data_process/boogu_image_ocr.py).

## Launch

```bash
bash examples/diffusionnft_trainer/boogu_image/run_boogu_image_diffusionnft_lora.sh
```

## Notes

- **Two LoRA adapters** (`policy_state_adapters=['default','old']`): rollout
  samples with `old`; training runs three forwards per step (policy grad,
  `old` no-grad, adapters-disabled reference). `old` refreshes by EMA
  (`algorithm.old_policy_decay_schedule`, `old_policy_update_interval`).
- Training forwards are CFG-free; rollout applies standard Boogu text CFG
  (`pipeline.guidance_scale`, upstream default 4.0).
- `algorithm.timestep_fraction` subsamples the rollout denoise schedule for
  training steps.
- The Boogu velocity convention (`x0 - noise`) is negated inside the
  training adapter to match the loss's diffusers-convention reconstruction —
  do not "fix" the sign in the loss.
- The vllm-omni `BooguImagePipeline` supports no TP/SP/CFG-parallel;
  `fsdp_layer_prefixes` must list Boogu's five block groups (see the launch
  script) for LoRA weight sync.
