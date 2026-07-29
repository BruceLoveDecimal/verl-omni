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
"""CPU tests for VisualRewardManager custom reward function kwargs."""

import os
from unittest.mock import MagicMock

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from verl import DataProto

from verl_omni.reward_loop.reward_manager.visual import VisualRewardManager


def reward_with_sampling_params(data_source, solution_image, ground_truth, extra_info, sampling_params=None):
    """Reward function that echoes back the sampling params it was called with."""
    return {"score": 1.0, "sampling_params": dict(sampling_params or {})}


def reward_without_sampling_params(data_source, solution_image, ground_truth, extra_info):
    """Reward function that does not declare ``sampling_params``."""
    return 0.25


def _make_config(custom_reward_function: dict):
    with initialize_config_dir(config_dir=os.path.abspath("verl_omni/trainer/config"), version_base=None):
        config = compose(config_name="diffusion_trainer")

    config.reward.custom_reward_function = OmegaConf.create(custom_reward_function)
    config.reward.reward_model.enable = False
    return config


def _make_manager(custom_reward_function: dict, compute_score) -> VisualRewardManager:
    config = _make_config(custom_reward_function)
    return VisualRewardManager(config, MagicMock(), compute_score)


def _make_single_data() -> DataProto:
    return DataProto.from_dict(
        tensors={"responses": torch.randn(1, 3, 64, 64)},
        non_tensors={
            "data_source": ["test_source"],
            "reward_model": [{"ground_truth": "hello"}],
            "extra_info": [{}],
        },
    )


class TestCustomRewardKwargs:
    def test_extra_config_fields_are_forwarded(self):
        """Fields beyond path/name reach a reward function that declares them."""
        manager = _make_manager(
            {"path": None, "name": "reward_with_sampling_params", "sampling_params": {"temperature": 0.0}},
            reward_with_sampling_params,
        )

        assert set(manager.custom_reward_kwargs) == {"sampling_params"}

        result = manager.loop.run_until_complete(manager.run_single(_make_single_data()))

        assert result["reward_score"] == pytest.approx(1.0)
        assert result["reward_extra_info"]["sampling_params"] == {"temperature": 0.0}

    def test_undeclared_fields_are_dropped(self):
        """A reward function that does not declare a field is not passed it."""
        manager = _make_manager(
            {"path": None, "name": "reward_without_sampling_params", "sampling_params": {"temperature": 0.0}},
            reward_without_sampling_params,
        )

        assert manager.custom_reward_kwargs == {}

        result = manager.loop.run_until_complete(manager.run_single(_make_single_data()))

        assert result["reward_score"] == pytest.approx(0.25)

    def test_path_and_name_are_never_forwarded(self):
        manager = _make_manager(
            {"path": "some/path.py", "name": "reward_with_sampling_params"},
            reward_with_sampling_params,
        )

        assert manager.custom_reward_kwargs == {}

        result = manager.loop.run_until_complete(manager.run_single(_make_single_data()))

        assert result["reward_extra_info"]["sampling_params"] == {}
