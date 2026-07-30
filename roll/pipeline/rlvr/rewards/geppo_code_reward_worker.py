"""GePPO-compatible reward worker for code-generation tasks."""

import json
from typing import Any, Dict

import torch

from roll.distributed.executor.worker import Worker
from roll.distributed.scheduler.decorator import Dispatch, register
from roll.distributed.scheduler.protocol import DataProto
from roll.models.model_providers import default_tokenizer_provider
from roll.pipeline.rlvr.rlvr_config import RewardConfig
from roll.utils.geppo_code_reward import compute_score


def _parse_ground_truth(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "item") and not isinstance(value, str):
        value = value.item()
    if isinstance(value, str):
        parsed = json.loads(value)
        if isinstance(parsed, dict):
            return parsed
    raise TypeError(f"Expected a JSON object for GePPO ground truth, got {type(value).__name__}")


class GeppoCodeRewardWorker(Worker):
    """Evaluate ROLL responses with GePPO's code reward semantics."""

    def __init__(self, worker_config: RewardConfig):
        super().__init__(worker_config=worker_config)
        self.rank_info.dp_rank = self.rank_info.rank
        self.rank_info.dp_size = self.rank_info.world_size
        self.tokenizer = default_tokenizer_provider(model_args=worker_config.model_args)
        self.livecodebench_data_dir = worker_config.livecodebench_data_dir
        self.timeout = worker_config.code_reward_timeout
        self.binary_reward = worker_config.code_reward_binary
        self.power4_reward = worker_config.code_reward_power4

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def initialize(self, pipeline_config):
        self.pipeline_config = pipeline_config

    @register(dispatch_mode=Dispatch.DP_MP_COMPUTE, clear_cache=False)
    def compute_rewards(self, data: DataProto) -> DataProto:
        responses = self.tokenizer.batch_decode(
            data.batch["responses"],
            skip_special_tokens=True,
        )
        ground_truths = data.non_tensor_batch.get("geppo_ground_truth")
        if ground_truths is None:
            raise KeyError(
                "GeppoCodeRewardWorker requires 'geppo_ground_truth'; "
                "use dataset_type=geppo_code"
            )
        if len(responses) != len(ground_truths):
            raise ValueError(
                f"Response/ground-truth size mismatch: {len(responses)} != {len(ground_truths)}"
            )

        scores = []
        for response, encoded_ground_truth in zip(responses, ground_truths):
            try:
                ground_truth = _parse_ground_truth(encoded_ground_truth)
                score, _ = compute_score(
                    response,
                    ground_truth,
                    livecodebench_data_dir=self.livecodebench_data_dir,
                    timeout=self.timeout,
                    is_binary_reward=self.binary_reward,
                    is_power4_reward=self.power4_reward,
                )
                scores.append(float(score))
            except Exception:
                self.logger.exception("GePPO code reward failed; assigning a zero reward")
                scores.append(0.0)

        token_level_rewards = torch.zeros_like(data.batch["responses"], dtype=torch.float16)
        score_tensor = torch.tensor(scores, dtype=torch.float16)
        return DataProto.from_dict(
            tensors={
                "token_level_rewards": token_level_rewards,
                "response_level_rewards": score_tensor.clone(),
                "scores": score_tensor,
            }
        )
