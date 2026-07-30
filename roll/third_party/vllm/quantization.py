import weakref
from functools import partial
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.nn import Module, Parameter

from vllm import _custom_ops as ops
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig, QuantizeMethodBase
from vllm.model_executor.layers.quantization.experts_int8 import ExpertsInt8MoEMethod
from vllm.model_executor.parameter import ChannelQuantScaleParameter, ModelWeightParameter
from vllm.model_executor.utils import set_weight_attrs

from roll.utils.logging import get_logger


logger = get_logger()

ROLL_INT8_QUANTIZATION_METHOD = "roll_int8"
FP16_QUANTIZATION_ALIASES = {"fp16", "float16"}


def normalize_inference_quantization(vllm_config: Dict[str, Any], dtype: str) -> Tuple[Dict[str, Any], str]:
    """Translate ROLL's user-facing quantization aliases to vLLM options."""
    quantization = vllm_config.get("quantization")
    if not isinstance(quantization, str):
        return vllm_config, dtype

    quantization = quantization.lower()
    if quantization in FP16_QUANTIZATION_ALIASES:
        vllm_config.pop("quantization")
        logger.info("Using FP16 inference weights via dtype=float16 (no vLLM quantization backend).")
        return vllm_config, "float16"
    if quantization == "int8":
        vllm_config["quantization"] = ROLL_INT8_QUANTIZATION_METHOD
        logger.info("Using ROLL online INT8 inference quantization.")
    return vllm_config, dtype


@torch.no_grad()
def per_channel_int8_quant(weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Symmetrically quantize a 2-D weight matrix by output channel."""
    if weight.ndim != 2:
        raise ValueError(f"INT8 linear weights must be 2-D, got shape {tuple(weight.shape)}")
    if not weight.is_floating_point():
        raise TypeError(f"INT8 online quantization expects floating-point weights, got {weight.dtype}")

    weight_fp32 = weight.float()
    max_abs = weight_fp32.abs().amax(dim=1, keepdim=True)
    scale = max_abs / torch.iinfo(torch.int8).max
    safe_scale = torch.where(max_abs > 0, scale, torch.ones_like(scale))
    quantized = torch.round(weight_fp32 / safe_scale).clamp(-127, 127).to(torch.int8)
    return quantized, safe_scale


def _int8_linear_weight_loader(
    layer_ref: weakref.ReferenceType,
    original_weight_loader,
    param: torch.Tensor,
    loaded_weight: torch.Tensor,
    *args,
    **kwargs,
) -> None:
    layer = layer_ref()
    if layer is None:
        raise RuntimeError("INT8 target layer was released while loading weights")
    if param is not layer.weight:
        raise ValueError("INT8 weight loader received an unexpected parameter")

    target_device = layer.weight.device
    quantized, scale = per_channel_int8_quant(loaded_weight.to(target_device))

    # vLLM loaders expect [output, input], while CUTLASS consumes [input, output].
    weight = ModelWeightParameter(
        data=layer.weight.data.t(),
        input_dim=1,
        output_dim=0,
        weight_loader=original_weight_loader,
    )
    weight_scale = ChannelQuantScaleParameter(
        data=layer.weight_scale.data,
        output_dim=0,
        weight_loader=original_weight_loader,
    )
    original_weight_loader(weight, quantized, *args, **kwargs)
    original_weight_loader(weight_scale, scale, *args, **kwargs)


class RollInt8LinearMethod(LinearMethodBase):
    def create_weights(
        self,
        layer: Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        del input_size, output_size, params_dtype
        original_weight_loader = extra_weight_attrs.get("weight_loader")
        if original_weight_loader is None:
            raise ValueError("ROLL INT8 requires vLLM's weight_loader attribute")

        output_size_per_partition = sum(output_partition_sizes)
        # CUTLASS consumes B as [input, output] with column-major strides.
        # Allocate [output, input] contiguously, then expose its transposed view.
        weight = Parameter(
            torch.empty(output_size_per_partition, input_size_per_partition, dtype=torch.int8).t(),
            requires_grad=False,
        )
        weight.weight_loader = partial(_int8_linear_weight_loader, weakref.ref(layer), original_weight_loader)
        layer.register_parameter("weight", weight)
        layer.register_parameter(
            "weight_scale",
            Parameter(torch.empty(output_size_per_partition, 1, dtype=torch.float32), requires_grad=False),
        )
        layer.logical_widths = output_partition_sizes

    def apply(self, layer: Module, x: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        input_2d = x.reshape(-1, x.shape[-1])
        quantized_input, input_scale, _ = ops.scaled_int8_quant(
            input_2d,
            scale=None,
            azp=None,
            symmetric=True,
        )
        output = ops.cutlass_scaled_mm(
            quantized_input,
            layer.weight,
            scale_a=input_scale,
            scale_b=layer.weight_scale,
            out_dtype=x.dtype,
            bias=bias,
        )
        return output.reshape(*x.shape[:-1], layer.weight.shape[1])


class RollInt8MoEMethod(ExpertsInt8MoEMethod):
    def create_weights(self, layer: Module, *args, **kwargs) -> None:
        super().create_weights(layer, *args, **kwargs)
        set_weight_attrs(layer.w13_weight, {"roll_skip_patch_moe": True})
        set_weight_attrs(layer.w2_weight, {"roll_skip_patch_moe": True})


class RollInt8Config(QuantizationConfig):
    def __init__(self) -> None:
        super().__init__()

    @classmethod
    def get_name(cls) -> str:
        return ROLL_INT8_QUANTIZATION_METHOD

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    @staticmethod
    def get_config_filenames() -> List[str]:
        return []

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "RollInt8Config":
        del config
        return cls()

    def get_quant_method(self, layer: Module, prefix: str) -> Optional[QuantizeMethodBase]:
        del prefix
        if isinstance(layer, LinearBase):
            return RollInt8LinearMethod()
        if isinstance(layer, FusedMoE):
            return RollInt8MoEMethod(self)
        return None


register_quantization_config(ROLL_INT8_QUANTIZATION_METHOD)(RollInt8Config)
