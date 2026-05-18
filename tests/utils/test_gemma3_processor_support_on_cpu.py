# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import types

import pytest
import torch

from verl.experimental.agent_loop.agent_loop import AgentLoopWorker
from verl.utils.model import compute_position_id_with_mask
from verl.utils.tokenizer import hf_processor


@pytest.mark.parametrize(
    ("processor_class_name", "model_type"), [("Gemma3Processor", "gemma3"), ("Gemma4Processor", "gemma4")]
)
def test_hf_processor_accepts_gemma_without_rope(monkeypatch, processor_class_name, model_type) -> None:
    import transformers

    processor = type(processor_class_name, (), {})()
    config = types.SimpleNamespace(model_type=model_type)

    monkeypatch.setattr(transformers.AutoProcessor, "from_pretrained", lambda *args, **kwargs: processor)
    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", lambda *args, **kwargs: config)

    loaded_processor = hf_processor(f"dummy-{model_type}")

    assert loaded_processor is processor
    assert loaded_processor.config is config
    assert not hasattr(loaded_processor, "get_rope_index")


def test_agent_loop_position_ids_fallback_without_rope() -> None:
    worker = AgentLoopWorker.__new__(AgentLoopWorker)
    worker.processor = types.SimpleNamespace()
    attention_mask = torch.tensor([[0, 1, 1, 1]], dtype=torch.long)
    input_ids = torch.tensor([[0, 10, 11, 12]], dtype=torch.long)

    position_ids = worker._compute_position_ids(input_ids, attention_mask, {})

    assert torch.equal(position_ids, compute_position_id_with_mask(attention_mask))
