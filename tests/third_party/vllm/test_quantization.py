import weakref

import pytest
import torch

from vllm.model_executor.layers.quantization import get_quantization_config

import roll.third_party.vllm.quantization as quantization_module
from roll.third_party.vllm.quantization import (
    ROLL_INT8_QUANTIZATION_METHOD,
    RollInt8LinearMethod,
    _int8_linear_weight_loader,
    normalize_inference_quantization,
    per_channel_int8_quant,
)


@pytest.mark.parametrize("alias", ["fp16", "float16", "FP16"])
def test_fp16_quantization_alias_uses_dtype(alias):
    config, dtype = normalize_inference_quantization({"quantization": alias, "block_size": 16}, "bfloat16")

    assert dtype == "float16"
    assert "quantization" not in config
    assert config["block_size"] == 16


def test_int8_quantization_alias_uses_roll_backend():
    config, dtype = normalize_inference_quantization({"quantization": "int8"}, "bfloat16")

    assert dtype == "bfloat16"
    assert config["quantization"] == ROLL_INT8_QUANTIZATION_METHOD
    assert get_quantization_config(ROLL_INT8_QUANTIZATION_METHOD).__name__ == "RollInt8Config"


def test_fp8_quantization_is_unchanged():
    config, dtype = normalize_inference_quantization({"quantization": "fp8"}, "bfloat16")

    assert config["quantization"] == "fp8"
    assert dtype == "bfloat16"


def test_per_channel_int8_quant_round_trip_and_zero_channel():
    weight = torch.tensor([[0.0, 0.0, 0.0], [-2.0, -0.25, 1.5], [0.1, 0.2, 0.3]])

    quantized, scale = per_channel_int8_quant(weight)
    reconstructed = quantized.float() * scale

    assert quantized.dtype == torch.int8
    assert scale.dtype == torch.float32
    assert scale.shape == (3, 1)
    assert torch.equal(quantized[0], torch.zeros(3, dtype=torch.int8))
    assert torch.isfinite(scale).all()
    assert torch.all((reconstructed - weight).abs() <= scale / 2 + 1e-6)


def test_int8_weight_loader_updates_transposed_weight_and_scale():
    class Layer:
        pass

    layer = Layer()
    layer.weight = torch.nn.Parameter(torch.empty(2, 3, dtype=torch.int8).t(), requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(torch.empty(2, 1, dtype=torch.float32), requires_grad=False)

    def copy_loader(param, loaded_weight):
        param.data.copy_(loaded_weight)

    loaded_weight = torch.tensor([[1.0, -2.0, 0.5], [0.0, 0.25, -0.5]], dtype=torch.bfloat16)
    _int8_linear_weight_loader(weakref.ref(layer), copy_loader, layer.weight, loaded_weight)

    reconstructed = layer.weight.t().float() * layer.weight_scale
    assert layer.weight.dtype == torch.int8
    assert layer.weight.stride() == (1, 3)
    assert torch.all((reconstructed - loaded_weight.float()).abs() <= layer.weight_scale / 2 + 1e-6)


def test_int8_linear_method_creates_cutlass_layout():
    layer = torch.nn.Module()

    def copy_loader(param, loaded_weight):
        param.data.copy_(loaded_weight)

    RollInt8LinearMethod().create_weights(
        layer=layer,
        input_size_per_partition=3,
        output_partition_sizes=[2, 4],
        input_size=3,
        output_size=6,
        params_dtype=torch.bfloat16,
        weight_loader=copy_loader,
    )

    assert layer.weight.shape == (3, 6)
    assert layer.weight.stride() == (1, 3)
    assert not layer.weight.is_contiguous()
    assert layer.weight.dtype == torch.int8
    assert layer.weight_scale.shape == (6, 1)
    assert callable(layer.weight.weight_loader)
    assert layer.logical_widths == [2, 4]


def test_int8_linear_apply_uses_dynamic_activation_quantization(monkeypatch):
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(torch.ones(2, 3, dtype=torch.int8).t(), requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(torch.ones(2, 1, dtype=torch.float32), requires_grad=False)
    calls = {}

    def fake_scaled_int8_quant(input_tensor, scale, azp, symmetric):
        calls["input_shape"] = input_tensor.shape
        assert scale is None
        assert azp is None
        assert symmetric is True
        return input_tensor.to(torch.int8), torch.ones(input_tensor.shape[0], 1), None

    def fake_cutlass_scaled_mm(a, b, scale_a, scale_b, out_dtype, bias):
        calls["weight_shape"] = b.shape
        calls["scale_shapes"] = (scale_a.shape, scale_b.shape)
        assert b.stride(0) == 1
        assert out_dtype == torch.float16
        assert bias is None
        return torch.zeros(a.shape[0], b.shape[1], dtype=out_dtype)

    monkeypatch.setattr(quantization_module.ops, "scaled_int8_quant", fake_scaled_int8_quant)
    monkeypatch.setattr(quantization_module.ops, "cutlass_scaled_mm", fake_cutlass_scaled_mm)

    output = RollInt8LinearMethod().apply(layer, torch.ones(2, 4, 3, dtype=torch.float16))

    assert output.shape == (2, 4, 2)
    assert calls == {
        "input_shape": torch.Size([8, 3]),
        "weight_shape": torch.Size([3, 2]),
        "scale_shapes": (torch.Size([8, 1]), torch.Size([2, 1])),
    }


def test_int8_quant_rejects_non_float_or_non_matrix_weights():
    with pytest.raises(TypeError, match="floating-point"):
        per_channel_int8_quant(torch.ones(2, 2, dtype=torch.int8))
    with pytest.raises(ValueError, match="2-D"):
        per_channel_int8_quant(torch.ones(2, 2, 2))
