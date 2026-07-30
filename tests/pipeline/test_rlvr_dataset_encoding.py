import json

import pytest

from roll.configs.data_args import DataArguments
from roll.utils import network_utils


# Importing the pipeline initializes the model download cache decorator, which
# only needs a stable node identifier for these source-level tests.
_original_get_node_ip = network_utils.get_node_ip
network_utils.get_node_ip = lambda: "127.0.0.1"
try:
    from roll.pipeline.rlvr.rlvr_pipeline import get_encode_function
finally:
    network_utils.get_node_ip = _original_get_node_ip


class RecordingTokenizer:
    def __init__(self):
        self.encoded_texts = None

    def apply_chat_template(self, conversation, tools=None, documents=None, **kwargs):
        return "chat:" + "|".join(message["content"] for message in conversation)

    def __call__(self, texts):
        self.encoded_texts = texts
        return {"input_ids": [[index] for index, _ in enumerate(texts)]}


def test_encode_function_auto_detects_default_messages_column():
    tokenizer = RecordingTokenizer()
    encode = get_encode_function("qwen2_5", tokenizer, DataArguments())
    messages = [[{"role": "user", "content": "one"}], [{"role": "user", "content": "two"}]]

    result = encode({"messages": [json.dumps(value) for value in messages], "prompt": ["raw one", "raw two"]})

    assert tokenizer.encoded_texts == ["chat:one", "chat:two"]
    assert result["input_ids"] == [[0], [1]]


def test_encode_function_auto_detects_prompt_when_messages_are_absent():
    tokenizer = RecordingTokenizer()
    encode = get_encode_function("qwen2_5", tokenizer, DataArguments())

    encode({"prompt": ["raw one", "raw two"]})

    assert tokenizer.encoded_texts == ["raw one", "raw two"]


def test_encode_function_honors_explicit_prompt_column():
    tokenizer = RecordingTokenizer()
    encode = get_encode_function("qwen2_5", tokenizer, DataArguments(prompt="question"))

    encode(
        {
            "messages": [[{"role": "user", "content": "should not be selected"}]],
            "question": ["selected prompt"],
        }
    )

    assert tokenizer.encoded_texts == ["selected prompt"]


def test_encode_function_reports_missing_explicit_column():
    tokenizer = RecordingTokenizer()
    encode = get_encode_function("qwen2_5", tokenizer, DataArguments(messages="conversation"))

    with pytest.raises(ValueError, match="Configured messages column 'conversation' is missing"):
        encode({"messages": [[{"role": "user", "content": "not the configured column"}]]})
