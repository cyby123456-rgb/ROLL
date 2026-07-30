"""Helpers for online FP8 weight updates in patched SGLang engines."""

from __future__ import annotations

import logging
from typing import Any, Optional, Tuple

import torch
from torch.nn import Parameter

try:
    from sglang.srt.layers.quantization.fp8_kernel import fp8_dtype, per_token_group_quant_fp8
except Exception:  # pragma: no cover - SGLang without FP8 support.
    fp8_dtype = None
    per_token_group_quant_fp8 = None

logger = logging.getLogger(__name__)


def _is_fp8_tensor(tensor: torch.Tensor) -> bool:
    return tensor.dtype in {torch.float8_e4m3fn, getattr(torch, "float8_e4m3fnuz", torch.float8_e4m3fn)}


def _ensure_channel_scale(layer: Any) -> None:
    """SGLang may create a scalar FP8 weight scale on sm80; online shard updates need per-channel scales."""
    weight_scale = getattr(layer, "weight_scale", None)
    weight = getattr(layer, "weight", None)
    if weight is None or weight_scale is None:
        return
    if not isinstance(weight_scale, torch.Tensor):
        return
    out_features = weight.shape[1]
    if weight_scale.numel() == out_features:
        return
    if weight_scale.numel() != 1:
        logger.warning(
            "Unexpected FP8 weight_scale shape for %s: %s, expected scalar or %s channels",
            getattr(layer, "prefix", type(layer).__name__),
            tuple(weight_scale.shape),
            out_features,
        )
        return
    expanded = torch.full(
        (1, out_features),
        float(weight_scale.detach().float().reshape(-1)[0].item()),
        device=weight.device,
        dtype=torch.float32,
    )
    layer.weight_scale = Parameter(expanded, requires_grad=False)


def _column_shard_range(layer: Any, loaded_shard_id: Any, loaded_weight: torch.Tensor) -> Tuple[int, torch.Tensor]:
    """Return target output-column offset and the local shard to quantize for fused column layers."""
    tp_rank = int(getattr(layer, "tp_rank", 0))
    tp_size = int(getattr(layer, "tp_size", 1))

    if loaded_shard_id in ("q", "k", "v") and all(
        hasattr(layer, attr) for attr in ("num_heads", "num_kv_heads", "head_size")
    ):
        q_size = int(layer.num_heads * layer.head_size)
        kv_size = int(layer.num_kv_heads * layer.head_size)
        if loaded_shard_id == "q":
            offset = 0
            shard_size = q_size
            start_idx = tp_rank * shard_size
        else:
            offset = q_size if loaded_shard_id == "k" else q_size + kv_size
            shard_size = kv_size
            replicas = int(getattr(layer, "num_kv_head_replicas", 1))
            shard_id = tp_rank // replicas
            start_idx = shard_id * shard_size
        if loaded_weight.shape[0] != shard_size:
            loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
        return offset, loaded_weight

    if isinstance(loaded_shard_id, int) and hasattr(layer, "output_sizes"):
        output_sizes = [int(size) for size in layer.output_sizes]
        offset = sum(output_sizes[:loaded_shard_id]) // tp_size
        shard_size = output_sizes[loaded_shard_id] // tp_size
        start_idx = tp_rank * shard_size
        if loaded_weight.shape[0] != shard_size:
            loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
        return offset, loaded_weight

    return 0, loaded_weight


def _fp8_online_weight_loader(layer: Any, param: torch.Tensor, loaded_weight: torch.Tensor, loaded_shard_id: Optional[Any] = None) -> None:
    if per_token_group_quant_fp8 is None:
        raise RuntimeError("SGLang FP8 quantization helpers are unavailable")
    weight = layer.weight
    weight_scale = layer.weight_scale
    target_device = weight.device
    quant_device = target_device
    if quant_device.type != "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("SGLang FP8 online update requires CUDA quantization")
        quant_device = torch.device("cuda", torch.cuda.current_device())

    with torch.no_grad():
        loaded_weight = loaded_weight.to(device=quant_device).contiguous()
        offset, loaded_weight = _column_shard_range(layer, loaded_shard_id, loaded_weight)
        qweight, scale = per_token_group_quant_fp8(loaded_weight, loaded_weight.shape[-1])
        qweight = qweight.t().contiguous().to(device=target_device)
        scale = scale.t().contiguous().to(device=weight_scale.device)

        if loaded_shard_id is None and qweight.shape == weight.shape:
            weight.copy_(qweight)
            if weight_scale.shape == scale.shape:
                weight_scale.copy_(scale)
            elif weight_scale.numel() == scale.numel():
                weight_scale.view(-1).copy_(scale.view(-1))
            else:
                raise RuntimeError(
                    f"FP8 scale shape mismatch for {getattr(layer, 'prefix', type(layer).__name__)}: "
                    f"target={tuple(weight_scale.shape)} source={tuple(scale.shape)}"
                )
            return

        end = offset + qweight.shape[1]
        if end > weight.shape[1]:
            raise RuntimeError(
                f"FP8 shard range out of bounds for {getattr(layer, 'prefix', type(layer).__name__)}: "
                f"offset={offset}, width={qweight.shape[1]}, weight_shape={tuple(weight.shape)}"
            )
        weight[:, offset:end].copy_(qweight)
        if weight_scale.ndim == 2:
            weight_scale[:, offset:end].copy_(scale)
        else:
            weight_scale.view(-1)[offset:end].copy_(scale.view(-1))


def patch_sglang_fp8_weight_loader(model: torch.nn.Module) -> None:
    """Attach online-update weight loaders to SGLang FP8 Parameters after quantization.

    SGLang's W8A8 FP8 path replaces ModelWeightParameter with plain Parameter after
    initial loading, which drops `weight_loader`. ROLL's online model_update calls
    model.load_weights(), so fused QKV/MLP loaders need this attribute restored.
    """
    patched = 0
    for module in model.modules():
        weight = getattr(module, "weight", None)
        weight_scale = getattr(module, "weight_scale", None)
        if not isinstance(weight, torch.Tensor) or not isinstance(weight_scale, torch.Tensor):
            continue
        if not _is_fp8_tensor(weight) or hasattr(weight, "weight_loader"):
            continue
        _ensure_channel_scale(module)
        weight.weight_loader = lambda param, loaded_weight, loaded_shard_id=None, layer=module: _fp8_online_weight_loader(
            layer, param, loaded_weight, loaded_shard_id
        )
        patched += 1
    if patched:
        logger.info("Patched %d SGLang FP8 parameters with online weight loaders", patched)
