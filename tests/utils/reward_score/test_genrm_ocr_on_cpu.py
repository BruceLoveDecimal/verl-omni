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
"""CPU tests for the GRM OCR reward, focused on configurable sampling params."""

import pytest

from verl_omni.utils.reward_score.genrm_ocr import (
    DEFAULT_GRM_PROMPT,
    DEFAULT_SAMPLING_PARAMS,
    SAMPLING_PARAMS_EXTRA_INFO_KEY,
    _build_chat_request,
    _resolve_sampling_params,
)


class TestResolveSamplingParams:
    def test_defaults_when_no_override(self):
        assert _resolve_sampling_params() == DEFAULT_SAMPLING_PARAMS
        assert _resolve_sampling_params(None, {}) == DEFAULT_SAMPLING_PARAMS

    def test_does_not_mutate_defaults(self):
        original = dict(DEFAULT_SAMPLING_PARAMS)

        _resolve_sampling_params({"temperature": 0.0})

        assert DEFAULT_SAMPLING_PARAMS == original

    def test_config_override_is_merged_not_replaced(self):
        resolved = _resolve_sampling_params({"temperature": 0.0})

        assert resolved["temperature"] == 0.0
        assert resolved["top_p"] == DEFAULT_SAMPLING_PARAMS["top_p"]
        assert resolved["max_tokens"] == DEFAULT_SAMPLING_PARAMS["max_tokens"]

    def test_unknown_keys_are_forwarded(self):
        resolved = _resolve_sampling_params({"seed": 42, "repetition_penalty": 1.05})

        assert resolved["seed"] == 42
        assert resolved["repetition_penalty"] == 1.05

    def test_extra_info_overrides_config(self):
        resolved = _resolve_sampling_params(
            {"temperature": 0.5, "top_p": 0.9},
            {SAMPLING_PARAMS_EXTRA_INFO_KEY: {"temperature": 0.0}},
        )

        assert resolved["temperature"] == 0.0
        assert resolved["top_p"] == 0.9

    def test_ignores_unrelated_extra_info(self):
        resolved = _resolve_sampling_params(extra_info={"frame_interval": 4, "num_turns": None})

        assert resolved == DEFAULT_SAMPLING_PARAMS

    def test_non_mapping_override_raises(self):
        with pytest.raises(TypeError, match="must be a mapping"):
            _resolve_sampling_params("temperature=0.0")

        with pytest.raises(TypeError, match="must be a mapping"):
            _resolve_sampling_params(extra_info={SAMPLING_PARAMS_EXTRA_INFO_KEY: [("temperature", 0.0)]})

    def test_accepts_omegaconf_dictconfig(self):
        """Hydra passes reward config sub-trees as DictConfig, not dict."""
        omegaconf = pytest.importorskip("omegaconf")

        resolved = _resolve_sampling_params(omegaconf.OmegaConf.create({"temperature": 0.0, "max_tokens": 128}))

        assert resolved["temperature"] == 0.0
        assert resolved["max_tokens"] == 128
        assert resolved["top_p"] == DEFAULT_SAMPLING_PARAMS["top_p"]


class TestBuildChatRequest:
    def test_request_carries_model_prompt_and_sampling_params(self):
        request = _build_chat_request(
            image_base64="data:image/png;base64,AAAA",
            model_name="some/grm",
            sampling_params={"temperature": 0.0, "max_tokens": 16},
        )

        assert request["model"] == "some/grm"
        assert request["temperature"] == 0.0
        assert request["max_tokens"] == 16

        user_content = request["messages"][-1]["content"]
        assert user_content[0]["image_url"]["url"] == "data:image/png;base64,AAAA"
        assert user_content[1]["text"] == DEFAULT_GRM_PROMPT

    def test_default_sampling_params_are_used_end_to_end(self):
        request = _build_chat_request("img", "some/grm", _resolve_sampling_params())

        for key, value in DEFAULT_SAMPLING_PARAMS.items():
            assert request[key] == value
