# Train Boogu-Image with Online Diffusion-DPO

Online DPO post-training for
[Boogu-Image-0.1-Base](https://huggingface.co/Boogu/Boogu-Image-0.1-Base)
(text-to-image). Prerequisites (the `boogu-image` package,
`trust_remote_code`, data preparation) are identical to the
[FlowGRPO Boogu recipe](../../flowgrpo_trainer/boogu_image/README.md).

## How it works

- Rollout generates `rollout.n` candidates per prompt with the
  **deterministic ODE** (standard Boogu text CFG, `guidance_scale=4.0`) and
  ships the final `latents_clean`.
- The reward model scores candidates; the trainer pairs the top-vs-bottom
  sample per prompt (`algorithm.paired_preference=true`).
- `DPODiffusersFSDPEngine` forward-noises the pairs on a **dense
  train-resolution schedule with Boogu's time shift applied**, so the
  noising sigma distribution matches what the checkpoint was trained on.
- Training-time forwards are **CFG-free** on both policy and reference
  (standard Diffusion-DPO); the Boogu velocity is negated into the DPO
  target space (`target = noise - x0`).

## Launch

```bash
bash examples/dpo_trainer/boogu_image/run_boogu_image_online_dpo_lora.sh
```

Notes:

- `dpo_beta` is set to `500.0` here (config default `2000.0`); tune per
  reward scale.
- Edit (TI2I) DPO rollouts are not integrated; requests carrying reference
  images fail with a clear error.
- The same rollout-parallelism limits as FlowGRPO apply (no TP/SP).
