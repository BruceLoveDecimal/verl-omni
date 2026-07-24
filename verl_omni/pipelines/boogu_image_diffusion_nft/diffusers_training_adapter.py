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

"""Boogu-Image training adapter for DiffusionNFT.

The NFT engine (``NFTDiffusersFSDPEngine``) constructs the forward-noised
``xt`` itself in the diffusers convention and hands the adapter exactly the
same shapes as the DPO engine: ``latents`` is ``(B, C, H, W)`` and
``timesteps`` is ``(B,)`` scheduler-native (``sigma * N``). Both inherited
methods therefore apply verbatim:

- ``prepare_model_inputs`` (from :class:`BooguImageDPO`): maps the timestep
  to Boogu time ``t = 1 - sigma`` and builds CFG-free transformer inputs.
  Unlike the Qwen NFT adapter, Boogu keeps training forwards CFG-free — the
  loss runs three predictions (policy / old / ref) per step, and doubling
  each with a CFG pass buys nothing for the forward-process objective.
- ``forward`` (from :class:`BooguImageDPO`): negates the Boogu velocity into
  the diffusers convention. This is **required**: ``DiffusionNFTLoss``
  reconstructs ``x0_prediction = xt - t * v``, which assumes
  ``v ~ noise - x0``, while the Boogu transformer predicts ``x0 - noise``.
"""

from verl_omni.pipelines.boogu_image_dpo.diffusers_training_adapter import BooguImageDPO
from verl_omni.pipelines.model_base import DiffusionModelBase

__all__ = ["BooguImageDiffusionNFT"]


@DiffusionModelBase.register("BooguImagePipeline", algorithm="diffusion_nft")
class BooguImageDiffusionNFT(BooguImageDPO):
    """Forward-process Boogu-Image adapter used by DiffusionNFT.

    A pure re-registration of the DPO adapter under the ``diffusion_nft``
    algorithm key: checkpoint loading (``build_module`` via the canonical
    ``boogu-image`` package), the dense shifted schedule, the sigma/velocity
    convention bridges, and the processor preparation hook are all shared.
    """
