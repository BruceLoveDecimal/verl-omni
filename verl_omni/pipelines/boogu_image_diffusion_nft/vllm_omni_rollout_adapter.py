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

"""Boogu-Image rollout adapter for DiffusionNFT.

DiffusionNFT trains on the forward process from final clean latents, so the
rollout side is the DPO adapter's deterministic ODE loop (``latents_clean``
in the output's ``rl`` metadata) plus one extra transport field:
``train_timesteps``, the denoise schedule of this rollout in the
scheduler-native diffusers convention. ``DiffusionNFTLoss.prepare_actor_batch``
subsamples it by ``algorithm.timestep_fraction`` and ``NFTDiffusersFSDPEngine`` loops
training steps over it (building ``xt`` from ``t = timestep / N``).
"""

from collections.abc import Mapping

import torch
from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch

from verl_omni.pipelines.boogu_image_dpo.vllm_omni_rollout_adapter import BooguImageDPOPipeline
from verl_omni.pipelines.model_base import VllmOmniPipelineBase

__all__ = ["BooguImageDiffusionNFTPipeline"]


def expand_train_timesteps(timesteps: torch.Tensor, batch_size: int) -> torch.Tensor:
    """Expand a ``(T,)`` denoise schedule to per-sample ``(B, T)`` transport shape."""
    return timesteps.detach().to(device="cpu", dtype=torch.float32).unsqueeze(0).expand(batch_size, -1).clone()


@VllmOmniPipelineBase.register("BooguImagePipeline", algorithm="diffusion_nft")
class BooguImageDiffusionNFTPipeline(BooguImageDPOPipeline):
    """Rollout pipeline shipping clean latents + the denoise schedule for NFT."""

    supports_request_batch = False

    def forward(self, req: OmniDiffusionRequest | DiffusionRequestBatch) -> DiffusionOutput | list[DiffusionOutput]:
        result = super().forward(req)

        for output in result if isinstance(result, list) else [result]:
            envelope = output.output
            if not isinstance(envelope, Mapping):
                # Engine warm-up / dummy run — nothing to transport.
                continue
            rl = envelope.get("metadata", {}).get("rl") or {}
            latents_clean = rl.get("latents_clean")
            if latents_clean is None:
                continue
            # ``super().forward`` configured the per-request schedule on
            # ``self.scheduler`` immediately before the denoise loop; it still
            # holds this request's timesteps here.
            rl["train_timesteps"] = expand_train_timesteps(self.scheduler.timesteps, latents_clean.shape[0])
        return result
