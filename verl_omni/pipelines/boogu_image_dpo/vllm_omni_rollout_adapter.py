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

"""Boogu-Image vLLM-Omni rollout adapter for online Diffusion-DPO (T2I).

Online DPO needs final clean latents per generated image (the trainer pairs
top/bottom rewards per prompt group), not reverse trajectories. Generation is
the deterministic ODE with standard Boogu text CFG; the FlowGRPO adapter's
prompt encoding and scheduler bridge are reused verbatim.
"""

import torch
import torch.nn.functional as F
from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch

from verl_omni.pipelines.boogu_image_flow_grpo.common import (
    apply_boogu_text_cfg,
    boogu_timestep_from_scheduler,
    configure_boogu_sde_timesteps,
    get_boogu_freqs_cis,
)
from verl_omni.pipelines.boogu_image_flow_grpo.vllm_omni_rollout_adapter import BooguImagePipelineWithLogProb
from verl_omni.pipelines.model_base import VllmOmniPipelineBase

__all__ = ["BooguImageDPOPipeline"]


@VllmOmniPipelineBase.register("BooguImagePipeline", algorithm="dpo")
class BooguImageDPOPipeline(BooguImagePipelineWithLogProb):
    """Rollout pipeline returning DPO training tensors with generated images.

    Reuses the FlowGRPO adapter's pre-tokenised prompt handling and the
    sigma-convention scheduler bridge; replaces the SDE trajectory loop with
    a plain ODE loop and ships ``latents_clean`` for forward-process DPO
    training. Reference (Edit / TI2I) rollouts are out of scope for DPO and
    rejected by the base class's request parsing when images are present.
    """

    supports_request_batch = False

    def forward(self, req: DiffusionRequestBatch) -> DiffusionOutput:
        prompts = req.prompts

        # The engine warm-up / dummy run injects a placeholder reference image
        # because the Boogu pipeline advertises ``support_image_input``; it is a
        # T2I warm-up, so skip the Edit-only rejection for it.
        if not req.is_dummy_run():
            _, preprocessed_images = self._extract_reference_images(prompts)
            if any(image is not None for image in preprocessed_images):
                raise NotImplementedError(
                    "BooguImageDPOPipeline supports text-to-image only; Edit (TI2I) DPO rollouts are not integrated."
                )

        prompt_ids, prompt_mask, negative_prompt_ids, negative_prompt_mask = self._extract_prompt_ids(prompts)
        if isinstance(prompt_ids, list):
            prompt_ids = torch.tensor(prompt_ids, device=self.device)
        if isinstance(negative_prompt_ids, list):
            negative_prompt_ids = torch.tensor(negative_prompt_ids, device=self.device)
        if prompt_ids is None:
            # Engine warm-up / dummy run without a usable prompt.
            return DiffusionOutput(output=None)

        sampling_params = req.sampling_params
        height = sampling_params.height or self.default_sample_size * self.vae_scale_factor
        width = sampling_params.width or self.default_sample_size * self.vae_scale_factor
        num_inference_steps = sampling_params.num_inference_steps or 50
        max_sequence_length = sampling_params.max_sequence_length or 1280
        guidance_scale = sampling_params.guidance_scale if sampling_params.guidance_scale_provided else 4.0
        num_images_per_prompt = (
            sampling_params.num_outputs_per_prompt if sampling_params.num_outputs_per_prompt > 0 else 1
        )

        generator = sampling_params.generator
        if generator is None and sampling_params.seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(sampling_params.seed)

        batch_size = prompt_ids.shape[0] if prompt_ids.ndim == 2 else 1

        prompt_embeds, prompt_embeds_mask = self.encode_prompt(
            prompt_ids=prompt_ids,
            attention_mask=prompt_mask,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
        )
        do_cfg = guidance_scale > 1.0 and negative_prompt_ids is not None
        if do_cfg:
            negative_prompt_embeds, negative_prompt_embeds_mask = self.encode_prompt(
                prompt_ids=negative_prompt_ids,
                attention_mask=negative_prompt_mask,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
            )
        else:
            negative_prompt_embeds = None
            negative_prompt_embeds_mask = None

        height, width, ori_height, ori_width = self._resolve_output_size(height, width)

        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            self.transformer.in_channels,
            height,
            width,
            torch.float32,
            self.device,
            generator,
        )

        num_tokens = latents.shape[-2] * latents.shape[-1]
        configure_boogu_sde_timesteps(
            self.scheduler,
            num_inference_steps=num_inference_steps,
            num_tokens=num_tokens,
            device=self.device,
            shift_config=self._shift_config,
        )
        self.scheduler.set_begin_index(0)
        freqs_cis = get_boogu_freqs_cis(self.transformer.axes_dim_rope, self.transformer.axes_lens)
        num_train_timesteps = self.scheduler.config.num_train_timesteps

        # Deterministic ODE denoising with standard Boogu text CFG.
        for timestep_value in self.scheduler.timesteps:
            boogu_t = boogu_timestep_from_scheduler(timestep_value, num_train_timesteps)
            x = latents.to(prompt_embeds.dtype)
            noise_pred = self.predict(boogu_t, x, prompt_embeds, freqs_cis, prompt_embeds_mask, None)
            if do_cfg:
                negative_noise_pred = self.predict(
                    boogu_t, x, negative_prompt_embeds, freqs_cis, negative_prompt_embeds_mask, None
                )
                noise_pred = apply_boogu_text_cfg(noise_pred, negative_noise_pred, guidance_scale)
            latents, _, _, _ = self.scheduler.step(
                noise_pred.to(torch.float32).neg(),
                timestep_value,
                latents.to(torch.float32),
                noise_level=0.0,
                sde_type="sde",
                return_logprobs=False,
                return_dict=False,
            )
            latents = latents.to(torch.float32)

        latents_clean = latents.float()

        # Decode for reward scoring / validation.
        output_type = sampling_params.output_type or "pil"
        if output_type == "latent":
            image = latents
        else:
            decode_latents = latents.to(dtype=self.vae.dtype)
            if self.vae.config.scaling_factor is not None:
                decode_latents = decode_latents / self.vae.config.scaling_factor
            if self.vae.config.shift_factor is not None:
                decode_latents = decode_latents + self.vae.config.shift_factor
            image = self.vae.decode(decode_latents, return_dict=False)[0]
            if (ori_height, ori_width) != (height, width):
                image = F.interpolate(image, size=(ori_height, ori_width), mode="bilinear")

        return DiffusionOutput(
            output={
                "payload": {"image": image},
                "metadata": {
                    "prompt_embeddings": {
                        "prompt_embeds": prompt_embeds,
                        "prompt_embeds_mask": prompt_embeds_mask,
                        "negative_prompt_embeds": negative_prompt_embeds,
                        "negative_prompt_embeds_mask": negative_prompt_embeds_mask,
                    },
                    "latents_clean": latents_clean,
                },
            },
            to_cpu=True,
        )
