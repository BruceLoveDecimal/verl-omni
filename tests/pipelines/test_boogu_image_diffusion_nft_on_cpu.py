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

"""CPU unit tests for the Boogu-Image DiffusionNFT adapters.

Pins the contracts that a tiny-model GPU smoke cannot distinguish:

1. **Velocity sign.** ``DiffusionNFTLoss`` reconstructs
   ``x0_pred = xt - t * v`` assuming the diffusers convention
   ``v ~ noise - x0``; the Boogu transformer predicts ``x0 - noise``, so the
   training adapter's ``forward`` must negate. A random-weight smoke passes
   either way — only this test fails on a dropped negation.
2. **Registry resolution** for ``("BooguImagePipeline", "diffusion_nft")``
   on both the training and rollout sides.
3. **``train_timesteps`` transport shape** ``(B, T)`` from the rollout's
   ``(T,)`` schedule.
"""

import pytest
import torch

vllm_omni = pytest.importorskip("vllm_omni", reason="rollout adapters import vllm_omni")

from verl_omni.pipelines.boogu_image_diffusion_nft.diffusers_training_adapter import (  # noqa: E402
    BooguImageDiffusionNFT,
)
from verl_omni.pipelines.boogu_image_diffusion_nft.vllm_omni_rollout_adapter import (  # noqa: E402
    expand_train_timesteps,
)
from verl_omni.pipelines.model_base import DiffusionModelBase, VllmOmniPipelineBase  # noqa: E402


class _BooguVelocityStub(torch.nn.Module):
    """Mimics the canonical transformer: returns a plain tensor (no tuple)."""

    def __init__(self, prediction: torch.Tensor):
        super().__init__()
        self.prediction = prediction

    def forward(self, **kwargs):  # noqa: ANN003
        return self.prediction


def test_forward_negates_boogu_velocity_into_diffusers_convention():
    boogu_velocity = torch.randn(2, 4, 8, 8)
    module = _BooguVelocityStub(boogu_velocity)

    out = BooguImageDiffusionNFT.forward(module, model_config=None, model_inputs={})

    torch.testing.assert_close(out, boogu_velocity.neg())


def test_registry_resolves_diffusion_nft_pair():
    import verl_omni.pipelines  # noqa: F401  (trigger @register decorators)

    training_cls = DiffusionModelBase.get_class_by_name("BooguImagePipeline", "diffusion_nft")
    assert training_cls.__name__ == "BooguImageDiffusionNFT"

    rollout_cls = VllmOmniPipelineBase.get_class("BooguImagePipeline", "diffusion_nft")
    assert rollout_cls.__name__ == "BooguImageDiffusionNFTPipeline"
    assert rollout_cls.supports_request_batch is False


def test_expand_train_timesteps_shape_and_values():
    schedule = torch.tensor([1000.0, 750.0, 500.0, 250.0])
    expanded = expand_train_timesteps(schedule, batch_size=3)

    assert expanded.shape == (3, 4)
    assert expanded.device.type == "cpu"
    assert expanded.dtype == torch.float32
    for row in expanded:
        torch.testing.assert_close(row, schedule.float())
    # Transport tensors must own their storage (no expanded views).
    assert expanded.is_contiguous()
