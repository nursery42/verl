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

from types import SimpleNamespace

import torch

from verl.experimental.agent_loop.agent_loop import AgentLoopWorker
from verl.utils.chat_template import apply_chat_template
from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset
from verl.utils.model import compute_position_id_with_mask, extract_multi_modal_inputs
from verl.utils.tokenizer import hf_processor


def test_hf_processor_accepts_gemma3_without_rope(monkeypatch):
    import transformers

    class Gemma3Processor:
        pass

    processor = Gemma3Processor()
    config = object()

    monkeypatch.setattr(transformers.AutoProcessor, "from_pretrained", lambda *args, **kwargs: processor)
    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", lambda *args, **kwargs: config)

    assert hf_processor("dummy-gemma3") is processor
    assert processor.config is config
    assert not hasattr(processor, "get_rope_index")


def test_agent_loop_worker_uses_text_position_ids_for_processors_without_rope():
    worker = object.__new__(AgentLoopWorker)
    worker.processor = object()

    attention_mask = torch.tensor([[1, 1, 0, 0]])
    position_ids = worker._compute_position_ids(
        input_ids=torch.tensor([[10, 11, 0, 0]]),
        attention_mask=attention_mask,
        multi_modal_inputs={"token_type_ids": torch.tensor([[0, 1, 0, 0]])},
    )

    assert torch.equal(position_ids, compute_position_id_with_mask(attention_mask))


def test_agent_loop_worker_rebuilds_gemma3_token_type_ids_from_padded_input_ids():
    class DummyTokenizer:
        def decode(self, input_ids, skip_special_tokens=True):
            del input_ids, skip_special_tokens
            return "prompt"

    class DummyBatchFeature(dict):
        def convert_to_tensors(self, tensor_type):
            assert tensor_type == "pt"
            return self

    class DummyProcessor:
        image_token_id = 99

        def __call__(self, **kwargs):
            assert kwargs["images"] == ["image"]
            return DummyBatchFeature(
                {
                    "input_ids": torch.tensor([[1, 2]]),
                    "attention_mask": torch.tensor([[1, 1]]),
                    "token_type_ids": torch.tensor([[0, 0]]),
                    "pixel_values": torch.ones(1, 3, 2, 2),
                }
            )

    worker = object.__new__(AgentLoopWorker)
    worker.processor = DummyProcessor()
    worker.tokenizer = DummyTokenizer()

    multi_modal_inputs = worker._compute_multi_modal_inputs(
        SimpleNamespace(multi_modal_data={"images": ["image"]}, mm_processor_kwargs={}),
        input_ids=torch.tensor([[0, 99, 99, 5, 0]]),
    )

    assert "input_ids" not in multi_modal_inputs
    assert "attention_mask" not in multi_modal_inputs
    assert multi_modal_inputs["token_type_ids"].tolist() == [[0, 1, 1, 0, 0]]


def test_apply_chat_template_trims_token_type_ids_after_dummy_prefix():
    class DummyProcessor:
        def __init__(self):
            self.calls = 0

        def apply_chat_template(self, messages, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("requires a user message")

            length = 2 if len(messages) == 1 else 5
            return {
                "input_ids": torch.arange(length).unsqueeze(0),
                "attention_mask": torch.ones(1, length, dtype=torch.long),
                "token_type_ids": torch.arange(10, 10 + length).unsqueeze(0),
            }

    output = apply_chat_template(
        DummyProcessor(),
        [{"role": "assistant", "content": [{"type": "text", "text": "hello"}]}],
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
    )

    assert output["input_ids"].tolist() == [[2, 3, 4]]
    assert output["attention_mask"].tolist() == [[1, 1, 1]]
    assert output["token_type_ids"].tolist() == [[12, 13, 14]]


def test_multiturn_sft_dataset_keeps_and_pads_token_type_ids(monkeypatch):
    import verl.utils.dataset.multiturn_sft_dataset as multiturn_sft_dataset

    class DummyIloc:
        def __getitem__(self, index):
            return self

        def to_dict(self):
            return {}

    class DummyTokenizer:
        pad_token_id = 0

    dataset = object.__new__(MultiTurnSFTDataset)
    dataset.dataframe = type("DummyDataFrame", (), {"iloc": DummyIloc()})()
    dataset.tools = None
    dataset.enable_thinking = None
    dataset.enable_thinking_default = None
    dataset.max_length = 6
    dataset.pad_mode = DatasetPadMode.RIGHT
    dataset.truncation = "right"
    dataset.tokenizer = DummyTokenizer()
    dataset.processor = None
    dataset.system_prompt = []
    dataset.generation_prompt = []
    dataset._build_messages = lambda _: [{"role": "user"}, {"role": "assistant"}]
    dataset.sanity_check = lambda *args, **kwargs: None

    def process_single_message(index, message, full_message, tools=None, enable_thinking=None):
        del message, full_message, tools, enable_thinking
        if index == 0:
            return (
                torch.tensor([10, 11]),
                torch.zeros(2, dtype=torch.long),
                torch.ones(2, dtype=torch.long),
                {"token_type_ids": torch.tensor([[0, 1]])},
            )
        return (
            torch.tensor([12, 13, 14]),
            torch.ones(3, dtype=torch.long),
            torch.ones(3, dtype=torch.long),
            {"token_type_ids": torch.tensor([[1, 1, 0]])},
        )

    dataset._process_single_message = process_single_message
    monkeypatch.setattr(multiturn_sft_dataset, "print_assembled_message", lambda *args, **kwargs: None)

    item = MultiTurnSFTDataset.__getitem__(dataset, 0)

    assert item["multi_modal_inputs"]["token_type_ids"].tolist() == [[0, 1, 1, 1, 0, 0]]


def test_extract_multi_modal_inputs_pads_variable_token_type_ids():
    multi_modal_inputs = extract_multi_modal_inputs(
        [
            {"token_type_ids": torch.tensor([[0, 1, 1]])},
            {"token_type_ids": torch.tensor([[0, 1]])},
        ]
    )

    assert multi_modal_inputs["token_type_ids"].tolist() == [[0, 1, 1], [0, 1, 0]]
