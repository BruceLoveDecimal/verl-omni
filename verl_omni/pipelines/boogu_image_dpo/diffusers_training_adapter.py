# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Boogu-Image training-side adapter for online Diffusion-DPO.

Consumed by ``DPODiffusersFSDPEngine`` (``model_type=diffusion_dpo_model``):
the engine forward-noises rollout ``latents_clean`` in the diffusers sigma
convention and compares the model prediction against ``target = noise - x0``
(``DPOLoss``). Two Boogu-specific bridges apply on top of the shared
FlowGRPO adapter:

- the Boogu transformer predicts ``x0 - noise`` — the **negative** of the
  DPO target space — so :meth:`forward` negates its output;
- ``sample_noise_and_timesteps`` indexes ``scheduler.timesteps`` with
  ``u * num_train_timesteps``, so the scheduler must carry the **dense**
  train-resolution schedule (not an inference-step schedule). We build it
  with Boogu's time shift applied so forward-noising matches the sigma
  distribution the checkpoint was trained on.
"""

from typing import Optional

import torch
from tensordict import TensorDict

from verl_omni.pipelines.boogu_image_flow_grpo.common import (
    boogu_timestep_from_scheduler,
    build_boogu_native_scheduler,
    configure_boogu_sde_timesteps,
    get_boogu_freqs_cis,
)
from verl_omni.pipelines.boogu_image_flow_grpo.diffusers_training_adapter import (
    BooguImage,
    _load_vae_scale_factor,
    _scheduler_num_train_timesteps,
)
from verl_omni.pipelines.model_base import DiffusionModelBase
from verl_omni.pipelines.schedulers import FlowMatchSDEDiscreteScheduler
from verl_omni.workers.config import DiffusionModelConfig

__all__ = ["BooguImageDPO"]


@DiffusionModelBase.register("BooguImagePipeline", algorithm="dpo")
class BooguImageDPO(BooguImage):
    """Training adapter for Boogu-Image online Diffusion-DPO (T2I).

    Inherits the FlowGRPO adapter's checkpoint loading, I2I condition hooks
    (T2I batches take the sentinel path), and output unwrapping. Training-time
    forwards are CFG-free on both policy and reference (standard
    Diffusion-DPO); rollout-side generation applies its own CFG.
    """

    @classmethod
    def build_scheduler(cls, model_config: DiffusionModelConfig):
        scheduler = FlowMatchSDEDiscreteScheduler.from_pretrained(
            pretrained_model_name_or_path=model_config.local_path,
            subfolder="scheduler",
        )
        cls.set_timesteps(scheduler, model_config, "cpu")
        return scheduler

    @classmethod
    def set_timesteps(cls, scheduler: FlowMatchSDEDiscreteScheduler, model_config: DiffusionModelConfig, device: str):
        """Configure the dense train-resolution shifted schedule.

        ``prepare_noisy_latents`` samples ``timesteps[(u * N).long()]`` with
        ``N = num_train_timesteps``, so the schedule length must be ``N``
        regardless of ``pipeline.num_inference_steps``.
        """
        vae_scale_factor = _load_vae_scale_factor(model_config.local_path)
        num_tokens = (int(model_config.pipeline.height) // vae_scale_factor) * (
            int(model_config.pipeline.width) // vae_scale_factor
        )
        configure_boogu_sde_timesteps(
            scheduler,
            native_scheduler=build_boogu_native_scheduler(model_config.local_path),
            num_inference_steps=int(scheduler.config.num_train_timesteps),
            num_tokens=num_tokens,
            device=device,
        )

    @classmethod
    def prepare_model_inputs(
        cls,
        module,
        model_config: DiffusionModelConfig,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        prompt_embeds: torch.Tensor,
        prompt_embeds_mask: torch.Tensor,
        negative_prompt_embeds: torch.Tensor,
        negative_prompt_embeds_mask: torch.Tensor,
        micro_batch: TensorDict,
        step: int,
    ) -> tuple[dict, Optional[dict]]:
        """Build transformer inputs for one forward-noised batch.

        Unlike the FlowGRPO adapter there is no trajectory dimension:
        ``latents`` is the already-noised ``(B, C, H, W)`` batch and
        ``timesteps`` is ``(B,)`` in the scheduler-native convention.
        Negative inputs are intentionally not built (CFG-free training).
        """
        del micro_batch, step, negative_prompt_embeds, negative_prompt_embeds_mask

        num_train_timesteps = _scheduler_num_train_timesteps(model_config.local_path)
        boogu_t = boogu_timestep_from_scheduler(timesteps, num_train_timesteps).to(latents.dtype)
        freqs_cis = get_boogu_freqs_cis(module.config.axes_dim_rope, module.config.axes_lens)

        model_inputs = {
            "hidden_states": latents,
            "timestep": boogu_t,
            "instruction_hidden_states": prompt_embeds,
            "freqs_cis": freqs_cis,
            "instruction_attention_mask": prompt_embeds_mask,
            "return_dict": False,
        }
        return model_inputs, None

    @classmethod
    def forward(
        cls,
        module,
        model_config: DiffusionModelConfig,
        model_inputs: dict[str, torch.Tensor],
        negative_model_inputs: Optional[dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Single prediction pass, negated into the DPO target space."""
        del negative_model_inputs
        noise_pred = super().forward(module, model_config, model_inputs)
        return noise_pred.neg()
