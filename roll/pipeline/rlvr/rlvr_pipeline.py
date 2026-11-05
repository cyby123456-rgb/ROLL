import copy
import json
import math
import os
import random
import shutil
import time
import uuid
from collections import defaultdict, deque
from datetime import datetime
from functools import partial
from typing import Any, Dict, List, Optional, Tuple

import datasets
import numpy as np
import ray
import torch
from codetiming import Timer
from torch.nn.utils.rnn import pad_sequence
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
from ray.util.timer import _Timer

from roll.configs import GeneratingArguments
from roll.datasets.chat_template import get_chat_template
from roll.datasets.collator import DataCollatorWithPaddingForPaddedKeys
from roll.datasets.dataset import get_dataset
from roll.distributed.executor.cluster import Cluster
from roll.distributed.scheduler.async_generate_scheduler import (
    AsyncDynamicSamplingScheduler,
    is_report_data_finished,
)
from roll.distributed.scheduler.generate_scheduler import DynamicSamplingScheduler, RequestScheduler
from roll.distributed.scheduler.protocol import (
    DataProto,
    TensorDict,
    make_non_tensor_array,
    repeat_non_tensor_value,
)
from roll.models.model_providers import default_tokenizer_provider
from roll.pipeline.base_pipeline import BasePipeline
from roll.pipeline.rlvr.rlvr_config import RLVRConfig
from roll.pipeline.rlvr.utils import dump_batch_to_reward_system, dump_rollout_to_specific_path
from roll.utils.dynamic_batching import dynamic_batching_shard
from roll.utils.functionals import (
    RunningMoments,
    agg_loss,
    compute_advantage,
    compute_reinforce_return,
    compute_token_reward,
    expand_to_token_level,
    get_sample_level_mask,
    masked_mean,
    reduce_metrics,
    masked_sum,
    reward_postprocess,
    concatenate_input_and_output,
    postprocess_generate,
)
from roll.utils.kl_controller import get_kl_controller
from roll.utils.logging import get_logger
from roll.utils.metrics.metrics_manager import MetricsManager


logger = get_logger()


def is_lora_training(pipeline_config: RLVRConfig) -> bool:
    return pipeline_config.actor_train.model_args.lora_target is not None


def preprocess_dataset(dataset, prompt_len, encode_function, data_args):
    # 处理数据
    print(f"Begin : {dataset}")
    dataset = dataset.map(
        encode_function,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        desc="Encoding dataset",
        load_from_cache_file=False,
    )
    # 过滤cutoff
    dataset = dataset.filter(
        lambda data_i: 5 < len(data_i["input_ids"]) <= prompt_len,
        num_proc=data_args.preprocessing_num_workers,
        desc="Filtering dataset",
    )
    print(f"Filtering prompt len: {dataset}")
    print(f"Encoding: {dataset}")
    return dataset


def get_encode_function(template_name, tokenizer, data_args):
    chat_template_func = get_chat_template(template_name, tokenizer)

    def encode_function(data_i):
        text_list = []
        configured_message_key = getattr(data_args, "messages", None)
        configured_prompt_key = getattr(data_args, "prompt", None)

        if configured_message_key is not None:
            if configured_message_key not in data_i:
                raise ValueError(
                    f"Configured messages column {configured_message_key!r} is missing from dataset columns "
                    f"{list(data_i.keys())}"
                )
            source_type, source_key = "messages", configured_message_key
        elif configured_prompt_key is not None:
            if configured_prompt_key not in data_i:
                raise ValueError(
                    f"Configured prompt column {configured_prompt_key!r} is missing from dataset columns "
                    f"{list(data_i.keys())}"
                )
            source_type, source_key = "prompt", configured_prompt_key
        elif "messages" in data_i:
            source_type, source_key = "messages", "messages"
        elif "prompt" in data_i:
            source_type, source_key = "prompt", "prompt"
        else:
            raise ValueError(
                "Dataset must contain a 'messages' or 'prompt' column when neither data_args.messages nor "
                f"data_args.prompt is configured; available columns: {list(data_i.keys())}"
            )

        if source_type == "messages":
            message_key = source_key
            for messages in data_i[message_key]:
                if isinstance(messages, str):
                    messages = json.loads(messages)
                text_list.append(chat_template_func(messages))
        else:
            prompt_key = source_key
            for prompt in data_i[prompt_key]:
                text_list.append(prompt)
        encodings = tokenizer(text_list)
        return encodings

    return encode_function


def update_dataset_domain(tag_2_domain: Dict[str, set[str]], row):
    if "domain" in row and row["domain"] is not None:
        return row
    row["domain"] = tag_2_domain.get(row["tag"], "math_rule")
    return row


def query_filter_fn(data_list: List[DataProto], config: RLVRConfig) -> bool:
    response_level_rewards = [data.batch["response_level_rewards"] for data in data_list]
    if len(response_level_rewards) == 1:
        return True

    rewards = torch.cat(response_level_rewards, dim=0)
    domain = data_list[0].non_tensor_batch["domain"][0]
    query_filter_config = config.rewards[domain].query_filter_config

    if query_filter_config.type == "mean_filter":
        threshold_up = query_filter_config.filter_args.get("threshold_up", math.inf)
        threshold_down = query_filter_config.filter_args.get("threshold_down", -1)
        mean_reward = torch.mean(rewards).item()
        return not (mean_reward <= threshold_down or mean_reward >= threshold_up)
    if query_filter_config.type == "std_filter":
        std_threshold = query_filter_config.filter_args.get("std_threshold", -1)
        return torch.std(rewards).item() > std_threshold
    return True


class RLVRPipeline(BasePipeline):

    def __init__(self, pipeline_config: RLVRConfig):
        super().__init__(pipeline_config)
        self.pipeline_config = pipeline_config
        self.use_ref_model = self.pipeline_config.enable_reference and (not is_lora_training(self.pipeline_config))
        scheduler_cls = (
            AsyncDynamicSamplingScheduler if self.pipeline_config.async_pipeline else DynamicSamplingScheduler
        )
        self.tokenizer = default_tokenizer_provider(model_args=self.pipeline_config.actor_train.model_args)

        self.lr_decay_for_baseline = bool(self.pipeline_config.lr_decay_for_baseline)
        self.lr_decay_for_baseline_decay_steps = int(self.pipeline_config.lr_decay_for_baseline_decay_steps)
        self.lr_decay_for_baseline_min_lr = float(self.pipeline_config.lr_decay_for_baseline_min_lr)
        self.baseline_current_actor_lr = float(self.pipeline_config.actor_train.training_args.learning_rate)
        if self.lr_decay_for_baseline:
            if self.lr_decay_for_baseline_decay_steps <= 0:
                raise ValueError("lr_decay_for_baseline_decay_steps must be > 0")
            if self.lr_decay_for_baseline_min_lr < 0:
                raise ValueError("lr_decay_for_baseline_min_lr must be >= 0")
            if self.lr_decay_for_baseline_min_lr > self.baseline_current_actor_lr:
                raise ValueError("lr_decay_for_baseline_min_lr must be <= actor_train.training_args.learning_rate")


        # ===== Update Validation (Scale Aligner) 基础初始化 =====
        # 初始化 update_validation_enabled（guard model 会在 strategy.initialize() 时创建）
        uv_config = self.pipeline_config.update_validation
        self.update_validation_enabled = uv_config.enable
        self.uv_metrics_only = bool(uv_config.metrics_only)
        self.uv_log_update_norm = uv_config.log_update_norm

        # ===== Random Rollback (与 UV 互斥；UV 优先) =====
        self.random_rollback_enable = bool(uv_config.random_rollback_enable) and not self.update_validation_enabled
        self.random_rollback_prob = float(uv_config.random_rollback_prob)
        if self.random_rollback_enable:
            if not (0.0 <= self.random_rollback_prob <= 1.0):
                raise ValueError(
                    f"update_validation.random_rollback_prob must be in [0, 1], got {self.random_rollback_prob}"
                )
            self._random_rollback_rng = random.Random(uv_config.random_rollback_seed)
            logger.info(
                f"Random Rollback enabled: prob={self.random_rollback_prob}, "
                f"seed={uv_config.random_rollback_seed}"
            )
        self.uv_lr_adaptation_enable = bool(uv_config.lr_adaptation_enable) and self.update_validation_enabled
        self.uv_lr_adaptation_window = int(uv_config.lr_adaptation_window)
        self.uv_lr_adaptation_interval = int(uv_config.lr_adaptation_interval)
        self.uv_lr_adaptation_high_rollback_rate = float(uv_config.lr_adaptation_high_rollback_rate)
        self.uv_lr_adaptation_low_rollback_rate = float(uv_config.lr_adaptation_low_rollback_rate)
        self.uv_lr_adaptation_decrease_factor = float(uv_config.lr_adaptation_decrease_factor)
        self.uv_lr_adaptation_increase_factor = float(uv_config.lr_adaptation_increase_factor)
        self.uv_lr_adaptation_min_lr = uv_config.lr_adaptation_min_lr
        self.uv_lr_adaptation_max_lr = uv_config.lr_adaptation_max_lr
        self.uv_lr_adaptation_last_step = -1
        self.uv_current_actor_lr = float(self.pipeline_config.actor_train.training_args.learning_rate)
        if uv_config.lr_adaptation_enable and not self.update_validation_enabled:
            logger.warning("update_validation.lr_adaptation_enable=True is ignored because update_validation.enable=False")
        if self.uv_lr_adaptation_enable:
            if self.uv_lr_adaptation_window <= 0:
                raise ValueError("update_validation.lr_adaptation_window must be > 0")
            if self.uv_lr_adaptation_interval <= 0:
                raise ValueError("update_validation.lr_adaptation_interval must be > 0")
            if not (0.0 <= self.uv_lr_adaptation_low_rollback_rate < self.uv_lr_adaptation_high_rollback_rate <= 1.0):
                raise ValueError(
                    "update_validation lr adaptation thresholds must satisfy "
                    "0 <= low < high <= 1"
                )
            if not (0.0 < self.uv_lr_adaptation_decrease_factor < 1.0):
                raise ValueError("update_validation.lr_adaptation_decrease_factor must be in (0, 1)")
            if self.uv_lr_adaptation_increase_factor <= 1.0:
                raise ValueError("update_validation.lr_adaptation_increase_factor must be > 1")
            if self.uv_lr_adaptation_min_lr is not None:
                self.uv_lr_adaptation_min_lr = float(self.uv_lr_adaptation_min_lr)
            if self.uv_lr_adaptation_max_lr is not None:
                self.uv_lr_adaptation_max_lr = float(self.uv_lr_adaptation_max_lr)
            if (
                self.uv_lr_adaptation_min_lr is not None
                and self.uv_lr_adaptation_max_lr is not None
                and self.uv_lr_adaptation_min_lr > self.uv_lr_adaptation_max_lr
            ):
                raise ValueError("update_validation.lr_adaptation_min_lr must be <= lr_adaptation_max_lr")

        # rollback 频率统计窗口（UV / random rollback 共用）
        if self.update_validation_enabled or self.random_rollback_enable:
            self.uv_rollback_history: deque = deque(maxlen=max(100, self.uv_lr_adaptation_window))

        if self.update_validation_enabled:
            # 初始化配置参数和 buffer（不依赖 domain_datasets）
            self.uv_batch_size = uv_config.batch_size
            self.uv_num_return_sequences = uv_config.num_return_sequences
            self.uv_generate_mode = str(getattr(uv_config, "generate_mode", "legacy"))
            if self.uv_generate_mode not in {"legacy", "scheduler"}:
                raise ValueError("update_validation.generate_mode must be 'legacy' or 'scheduler'")
            self.uv_generate_concurrency = int(getattr(uv_config, "generate_concurrency", 64))
            if self.uv_generate_concurrency < 1:
                raise ValueError("update_validation.generate_concurrency must be >= 1")
            self.uv_min_steps_for_threshold = uv_config.min_steps_for_threshold
            self.uv_buffer_size = uv_config.buffer_size
            self.uv_threshold_percentile = uv_config.advantage_threshold_percentile
            self.uv_acceptance_advantage_threshold = uv_config.acceptance_advantage_threshold
            self.uv_acceptance_advantage_threshold_mode = uv_config.acceptance_advantage_threshold_mode
            self.uv_acceptance_advantage_threshold_reverse_decay_start = (
                uv_config.acceptance_advantage_threshold_reverse_decay_start
            )
            self.uv_acceptance_advantage_threshold_reverse_decay_end = (
                uv_config.acceptance_advantage_threshold_reverse_decay_end
            )
            self.uv_acceptance_advantage_threshold_reverse_decay_steps = (
                uv_config.acceptance_advantage_threshold_reverse_decay_steps
            )
            self.uv_is_level = uv_config.is_level
            self.uv_data_mode = uv_config.data_mode  # "fixed" | "rollout_batch" | "random_per_step"
            self.uv_est_mode = uv_config.est_mode  # "on-policy" | "off-policy"
            self.uv_offpolicy_rollout_n = uv_config.offpolicy_rollout_n
            if self.uv_offpolicy_rollout_n < 1:
                raise ValueError("update_validation.offpolicy_rollout_n must be >= 1")
            if self.uv_acceptance_advantage_threshold_mode == "reverse_decay":
                if self.uv_acceptance_advantage_threshold_reverse_decay_steps <= 0:
                    raise ValueError(
                        "update_validation.acceptance_advantage_threshold_reverse_decay_steps must be > 0 "
                        "when acceptance_advantage_threshold_mode='reverse_decay'"
                    )
                if (
                    self.uv_acceptance_advantage_threshold_reverse_decay_end
                    < self.uv_acceptance_advantage_threshold_reverse_decay_start
                ):
                    raise ValueError(
                        "update_validation.acceptance_advantage_threshold_reverse_decay_end must be >= "
                        "acceptance_advantage_threshold_reverse_decay_start when mode='reverse_decay'"
                    )
            self.uv_offpolicy_history_maxlen = self.uv_offpolicy_rollout_n
            self.uv_advantage_buffer: deque = deque(maxlen=self.uv_buffer_size)
            self.uv_advantage_threshold: Optional[float] = None

        dataset_paths = []
        if self.pipeline_config.actor_train.data_args.file_name:
            train_file_names = self.pipeline_config.actor_train.data_args.file_name
            dataset_paths.extend(train_file_names if isinstance(train_file_names, list) else [train_file_names])

        print(f"load_dataset_paths: {chr(10)} {chr(10).join(dataset_paths)}")
        dataset = get_dataset(self.pipeline_config.actor_train.data_args)

        self.val_dataset = None
        if self.pipeline_config.validation and self.pipeline_config.validation.data_args:
            self.val_dataset = get_dataset(self.pipeline_config.validation.data_args)

        # 加上format，然后转ids的func
        template_name = (
            self.pipeline_config.global_template
            if self.pipeline_config.global_template
            else self.pipeline_config.actor_train.data_args.template
        )
        encode_function = get_encode_function(template_name, self.tokenizer, self.pipeline_config.actor_train.data_args)

        dataset = preprocess_dataset(
            dataset,
            self.pipeline_config.prompt_length,
            encode_function,
            data_args=self.pipeline_config.actor_train.data_args,
        )
        # update domain field
        dataset = dataset.map(
            partial(update_dataset_domain, self.pipeline_config.tag_2_domain),
            num_proc=self.pipeline_config.actor_train.data_args.preprocessing_num_workers,
            desc="update_dataset_domain",
            load_from_cache_file=False,
        )
        self.domain_datasets: Dict[str, datasets.Dataset] = {}
        for domain in self.pipeline_config.actor_train.data_args.domain_interleave_probs.keys():
            self.domain_datasets[domain] = dataset.filter(
                lambda example, dom: example["domain"] == dom,
                num_proc=self.pipeline_config.actor_train.data_args.preprocessing_num_workers,
                fn_kwargs={"dom": domain},
            )
            assert len(self.domain_datasets[domain]) > 0, f"domain dataset {domain} has no data"

        if self.val_dataset:
            validation_data_args = self.pipeline_config.validation.data_args
            validation_template_name = (
                self.pipeline_config.global_template
                if self.pipeline_config.global_template
                else validation_data_args.template
            )
            validation_encode_function = get_encode_function(
                validation_template_name, self.tokenizer, validation_data_args
            )
            self.val_dataset = preprocess_dataset(
                self.val_dataset,
                self.pipeline_config.prompt_length,
                validation_encode_function,
                data_args=validation_data_args,
            )
            self.val_dataset = self.val_dataset.map(
                partial(update_dataset_domain, self.pipeline_config.tag_2_domain),
                num_proc=self.pipeline_config.actor_train.data_args.preprocessing_num_workers,
                desc="update_val_dataset_domain",
                load_from_cache_file=False,
            )
            assert "domain" in self.val_dataset.column_names, "domain field should set in val dataset"

        assert "domain" in dataset.column_names, "domain field should set in dataset"
        print(dataset)

        # ===== Update Validation (Scale Aligner) 数据集初始化 =====
        # 在 domain_datasets 初始化之后，仅 data_mode=="fixed" 时预采样固定验证集
        if self.update_validation_enabled and self.uv_data_mode == "fixed":
            self._setup_validation_dataset()
            logger.info(
                f"Update Validation config: data_mode={self.uv_data_mode}, batch_size={self.uv_batch_size}, "
                f"num_return_sequences={self.uv_num_return_sequences}, "
                f"min_steps={self.uv_min_steps_for_threshold}, "
                f"buffer_size={self.uv_buffer_size}, "
                f"threshold_percentile={self.uv_threshold_percentile}, "
                f"acceptance_threshold={self.uv_acceptance_advantage_threshold}, "
                f"acceptance_threshold_mode={self.uv_acceptance_advantage_threshold_mode}, "
                f"acceptance_threshold_reverse_decay_start={self.uv_acceptance_advantage_threshold_reverse_decay_start}, "
                f"acceptance_threshold_reverse_decay_end={self.uv_acceptance_advantage_threshold_reverse_decay_end}, "
                f"acceptance_threshold_reverse_decay_steps={self.uv_acceptance_advantage_threshold_reverse_decay_steps}, "
                f"is_level={self.uv_is_level}, "
                f"est_mode={self.uv_est_mode}, "
                f"offpolicy_rollout_n={self.uv_offpolicy_rollout_n}, "
                f"generate_mode={self.uv_generate_mode}, "
                f"generate_concurrency={self.uv_generate_concurrency}, "
                f"log_update_norm={self.uv_log_update_norm}"
            )
        elif self.update_validation_enabled:
            logger.info(
                f"Update Validation config: data_mode={self.uv_data_mode}, batch_size={self.uv_batch_size}, "
                f"num_return_sequences={self.uv_num_return_sequences}, "
                f"min_steps={self.uv_min_steps_for_threshold}, "
                f"buffer_size={self.uv_buffer_size}, "
                f"threshold_percentile={self.uv_threshold_percentile}, "
                f"acceptance_threshold={self.uv_acceptance_advantage_threshold}, "
                f"acceptance_threshold_mode={self.uv_acceptance_advantage_threshold_mode}, "
                f"acceptance_threshold_reverse_decay_start={self.uv_acceptance_advantage_threshold_reverse_decay_start}, "
                f"acceptance_threshold_reverse_decay_end={self.uv_acceptance_advantage_threshold_reverse_decay_end}, "
                f"acceptance_threshold_reverse_decay_steps={self.uv_acceptance_advantage_threshold_reverse_decay_steps}, "
                f"is_level={self.uv_is_level}, "
                f"est_mode={self.uv_est_mode}, "
                f"offpolicy_rollout_n={self.uv_offpolicy_rollout_n}, "
                f"generate_mode={self.uv_generate_mode}, "
                f"generate_concurrency={self.uv_generate_concurrency}, "
                f"log_update_norm={self.uv_log_update_norm}"
            )

        self.kl_ctrl = get_kl_controller(
            init_kl_coef=self.pipeline_config.init_kl_coef,
            target_kl=self.pipeline_config.target_kl,
            kl_horizon=self.pipeline_config.kl_horizon,
        )

        assert self.pipeline_config.max_steps > 0, "max_steps must be greater than 0"
        self.pipeline_config.set_max_steps(max_steps=self.pipeline_config.max_steps)

        self.actor_train: Any = Cluster(
            name=self.pipeline_config.actor_train.name,
            worker_cls=self.pipeline_config.actor_train.worker_cls,
            resource_manager=self.resource_manager,
            worker_config=self.pipeline_config.actor_train,
        )
        self.actor_infer: Any = Cluster(
            name=self.pipeline_config.actor_infer.name,
            worker_cls=self.pipeline_config.actor_infer.worker_cls,
            resource_manager=self.resource_manager,
            worker_config=self.pipeline_config.actor_infer,
        )
        download_clusters = [self.actor_train, self.actor_infer]
        # use unwrapped model as reference for lora training
        if self.use_ref_model:
            self.reference: Any = Cluster(
                name=self.pipeline_config.reference.name,
                worker_cls=self.pipeline_config.reference.worker_cls,
                resource_manager=self.resource_manager,
                worker_config=self.pipeline_config.reference,
            )
            download_clusters.append(self.reference)
        if self.pipeline_config.adv_estimator == "gae":
            self.critic: Any = Cluster(
                name=self.pipeline_config.critic.name,
                worker_cls=self.pipeline_config.critic.worker_cls,
                resource_manager=self.resource_manager,
                worker_config=self.pipeline_config.critic,
            )
            download_clusters.append(self.critic)
        self.rewards: Dict[str, Any] = {
            key: Cluster(
                name=f"reward-{key}",
                worker_cls=worker_config.worker_cls,
                resource_manager=self.resource_manager,
                worker_config=worker_config,
            )
            for key, worker_config in self.pipeline_config.rewards.items()
        }
        download_clusters.extend(self.rewards.values())
        self.download_models(*download_clusters)

        domain_ratios = self.pipeline_config.actor_train.data_args.domain_interleave_probs
        self.generate_schedulers: Dict[str, DynamicSamplingScheduler] = {}
        self.domain_batch_size = {}
        domain_list = list(domain_ratios.keys())
        accumulated = 0
        for i, domain in enumerate(domain_list):
            if i == len(domain_list) - 1:
                domain_batch_size = self.pipeline_config.rollout_batch_size - accumulated
            else:
                domain_batch_size = int(domain_ratios[domain] * self.pipeline_config.rollout_batch_size)
            accumulated += domain_batch_size
            generate_scheduler = scheduler_cls.options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=ray.get_runtime_context().get_node_id(),
                    soft=False,
                )
            ).remote(pipeline_config=self.pipeline_config)
            ray.get(
                generate_scheduler.set_scheduler.remote(
                    actor_cluster=self.actor_infer,
                    reward_clusters={domain: self.rewards[domain]},
                    dataset=self.domain_datasets[domain],
                    collect_fn_cls=DataCollatorWithPaddingForPaddedKeys,
                    collect_fn_kwargs=dict(max_length=self.pipeline_config.prompt_length, padding="max_length"),
                    response_filter_fn=lambda data_item, config: True,
                    query_filter_fn=query_filter_fn,
                    response_callback_fn=generate_scheduler.report_response.remote,
                    state=self.state.kv.get(f"scheduler_state_{domain}", None),
                )
            )
            self.generate_schedulers[domain] = generate_scheduler
            self.domain_batch_size[domain] = domain_batch_size

            assert domain_batch_size < len(self.domain_datasets[domain]), (
                f"domain_batch_size {domain_batch_size} must be "
                f"less than the number of domain datasets {len(self.domain_datasets[domain])}"
            )

        if self.val_dataset:
            val_pipeline_config = copy.deepcopy(self.pipeline_config)
            val_pipeline_config.is_use_additional_prompts = False
            self.val_generate_scheduler = scheduler_cls.options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=ray.get_runtime_context().get_node_id(),
                    soft=False,
                )
            ).remote(pipeline_config=val_pipeline_config)
        if self.val_dataset:
            ray.get(
                self.val_generate_scheduler.set_scheduler.remote(
                    actor_cluster=self.actor_infer,
                    reward_clusters=self.rewards,
                    dataset=self.val_dataset,
                    collect_fn_cls=DataCollatorWithPaddingForPaddedKeys,
                    collect_fn_kwargs=dict(max_length=self.pipeline_config.prompt_length, padding="max_length"),
                    response_filter_fn=lambda data_item, config: True,
                    query_filter_fn=lambda data_list, config: True,
                    response_callback_fn=self.val_generate_scheduler.report_response.remote,
                    is_val=True,
                )
            )

        refs = []
        refs.extend(self.actor_infer.initialize(pipeline_config=self.pipeline_config, blocking=False))
        ray.get(refs)

        self.uv_request_scheduler = None
        if self.update_validation_enabled and self.uv_generate_mode == "scheduler":
            self.uv_request_scheduler = RequestScheduler.options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=ray.get_runtime_context().get_node_id(),
                    soft=False,
                ),
                # generate calls wait for callbacks, so reserve actor slots for
                # report_response and abort_request while the window is full.
                max_concurrency=self.uv_generate_concurrency + 2,
            ).remote(infer_cluster=self.actor_infer, pipeline_config=self.pipeline_config)

        if self.use_ref_model:
            refs.extend(self.reference.initialize(pipeline_config=self.pipeline_config, blocking=True))

        refs = []
        for key, cluster in self.rewards.items():
            refs.extend(cluster.initialize(pipeline_config=self.pipeline_config, blocking=False))
        ray.get(refs)

        refs: List[ray.ObjectRef] = []
        refs.extend(self.actor_train.initialize(pipeline_config=self.pipeline_config, blocking=False))
        if self.pipeline_config.adv_estimator == "gae":
            refs.extend(self.critic.initialize(pipeline_config=self.pipeline_config, blocking=False))
        ray.get(refs)

        self.set_model_update_pair(
            src_cluster=self.actor_train,
            tgt_cluster=self.actor_infer,
            frequency=self.pipeline_config.actor_train.model_update_frequency,
        )

        if self.pipeline_config.adv_estimator == "gae":
            self.set_checkpoint_clusters(self.actor_train, self.critic)
        else:
            self.set_checkpoint_clusters(self.actor_train)

        self.running = {}
        for domain in self.rewards.keys():
            self.running[domain] = RunningMoments()

        # cache rollout用于跨step对比策略收益
        self.prev_rollout_cache: Optional[DataProto] = None
        self.prev_policy_return: Optional[float] = None
        self.prev_rollout_step: Optional[int] = None
        if self.update_validation_enabled and self.uv_est_mode == "off-policy":
            self.uv_offpolicy_rollout_history: deque = deque(maxlen=self.uv_offpolicy_history_maxlen)

    # ===== Update Validation (Scale Aligner) Methods =====


    def _setup_validation_dataset(self):
        """设置验证用的数据集，从训练数据中采样固定的 prompts"""
        # 从各个 domain 数据集中采样
        total_samples_needed = self.uv_batch_size
        samples_per_domain = max(1, total_samples_needed // len(self.domain_datasets))

        # 打印每个domain dataset的columns，用于调试
        for domain, dataset in self.domain_datasets.items():
            logger.info(f"Update Validation: domain '{domain}' dataset columns = {dataset.column_names}")

        validation_samples = []
        required_fields = ["input_ids", "ground_truth"]  # reward 计算必需的字段

        for domain, dataset in self.domain_datasets.items():
            # 从每个 domain 采样
            n_samples = min(samples_per_domain, len(dataset))
            indices = np.random.choice(len(dataset), size=n_samples * 2, replace=False)  # 多采样一些以应对过滤

            collected = 0
            for idx in indices:
                if collected >= n_samples:
                    break
                sample = dataset[int(idx)]

                # 检查必需字段是否存在且非空
                has_required = all(
                    field in sample and sample[field] is not None
                    for field in required_fields
                )
                if not has_required:
                    logger.warning(f"Update Validation: skipping sample {idx} in domain '{domain}' - missing required fields")
                    continue

                sample["domain"] = domain
                validation_samples.append(sample)
                collected += 1

            if collected < n_samples:
                logger.warning(f"Update Validation: domain '{domain}' only collected {collected}/{n_samples} valid samples")

        # 截断到需要的数量
        self.uv_validation_samples = validation_samples[:total_samples_needed]

        # 验证采样结果是否包含必要字段
        if len(self.uv_validation_samples) > 0:
            sample_keys = set(self.uv_validation_samples[0].keys())
            logger.info(f"Update Validation: sample keys = {sample_keys}")
            if "ground_truth" not in sample_keys:
                logger.warning(f"Update Validation: 'ground_truth' not found in samples! Available keys: {sample_keys}")

        logger.info(f"Update Validation: prepared {len(self.uv_validation_samples)} validation samples")

    def _sample_validation_from_dataset(self) -> List[Dict]:
        """从各 domain 数据集中随机采样一批验证样本（与 _setup_validation_dataset 逻辑一致，但不写入 self）。
        用于 data_mode=\"random_per_step\" 时每步重新采样。
        """
        total_samples_needed = self.uv_batch_size
        samples_per_domain = max(1, total_samples_needed // len(self.domain_datasets))
        required_fields = ["input_ids", "ground_truth"]
        validation_samples = []
        for domain, dataset in self.domain_datasets.items():
            n_samples = min(samples_per_domain, len(dataset))
            indices = np.random.choice(len(dataset), size=n_samples * 2, replace=False)
            collected = 0
            for idx in indices:
                if collected >= n_samples:
                    break
                sample = dict(dataset[int(idx)])
                has_required = all(
                    field in sample and sample[field] is not None for field in required_fields
                )
                if not has_required:
                    continue
                sample["domain"] = domain
                validation_samples.append(sample)
                collected += 1
        return validation_samples[:total_samples_needed]

    def _prepare_validation_prompts_from_rollout_batch(self, rollout_batch: DataProto) -> DataProto:
        """用当前 step 的 rollout batch（更新前用于生成的那批 prompt）构造验证用 DataProto。
        用于 data_mode=\"rollout_batch\"：验证时用同一批 prompt 再生成一次算 advantage。
        """
        from roll.distributed.scheduler.protocol import TensorDict
        prompts = rollout_batch.batch["prompts"].clone()
        rollout_num_return_sequences = (
            rollout_batch.meta_info.get("generation_config", {}).get(
                "num_return_sequences",
                self.pipeline_config.actor_infer.generating_args.num_return_sequences,
            )
            if rollout_batch.meta_info is not None
            else self.pipeline_config.actor_infer.generating_args.num_return_sequences
        )
        rollout_num_return_sequences = int(rollout_num_return_sequences)

        # 训练主流程传进来的 rollout_batch 已经是 response-level batch。
        # 同一个 prompt 会按 rollout 的 num_return_sequences 连续展开。
        # 这里先还原到 prompt-level，再交给 validation generate 重新展开，
        # 使 rollout_batch mode 和 fixed mode 的输入语义一致。
        if rollout_num_return_sequences > 1:
            total_samples = prompts.shape[0]
            if total_samples % rollout_num_return_sequences != 0:
                raise ValueError(
                    "Update Validation data_mode=rollout_batch expects a response-level "
                    "rollout batch grouped by prompt, but got "
                    f"batch_size={total_samples} and "
                    f"num_return_sequences={rollout_num_return_sequences}"
                )
            prompt_level_indices = torch.arange(
                0, total_samples, rollout_num_return_sequences, device=prompts.device
            )
            prompts = prompts.index_select(0, prompt_level_indices).clone()
        else:
            prompt_level_indices = None
        pad_token_id = self.tokenizer.pad_token_id
        attention_mask = (prompts != pad_token_id).long()
        if attention_mask.dtype != torch.float32:
            attention_mask = attention_mask.float()
        position_ids = torch.clamp(torch.cumsum(attention_mask, dim=-1) - 1, min=0)
        batch = TensorDict(
            {
                "input_ids": prompts,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "prompts": prompts.clone(),
            },
            batch_size=prompts.shape[0],
        )
        if "domain" in rollout_batch.batch:
            domain_tensor = rollout_batch.batch["domain"]
            if prompt_level_indices is not None:
                domain_tensor = domain_tensor.index_select(0, prompt_level_indices)
            batch["domain"] = domain_tensor.clone()
        else:
            domain_list = rollout_batch.non_tensor_batch.get("domain", ["math_rule"] * rollout_batch.batch["prompts"].shape[0])
            if prompt_level_indices is not None:
                domain_list = [domain_list[int(i)] for i in prompt_level_indices.cpu().tolist()]
            batch["domain"] = torch.tensor([hash(d) % (2**31) for d in domain_list])
        non_tensor_batch = {}
        if rollout_batch.non_tensor_batch is not None:
            for k, v in rollout_batch.non_tensor_batch.items():
                values = make_non_tensor_array(v) if not isinstance(v, np.ndarray) else v
                if prompt_level_indices is not None:
                    values = values[prompt_level_indices.cpu().numpy()]
                non_tensor_batch[k] = values
        if "domain" not in non_tensor_batch and rollout_batch.non_tensor_batch and "domain" in rollout_batch.non_tensor_batch:
            domain_values = np.array(rollout_batch.non_tensor_batch["domain"], dtype=object)
            if prompt_level_indices is not None:
                domain_values = domain_values[prompt_level_indices.cpu().numpy()]
            non_tensor_batch["domain"] = domain_values
        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch or None, meta_info={})

    def _sync_to_guard(self, global_step: int):
        """同步 actor_train 的状态到 guard（保存状态）

        在同一个 worker 内部，直接调用 strategy 的方法同步模型和优化器状态。
        避免跨 worker 通信的复杂性。

        Args:
            global_step: 当前 step 编号
        """
        logger.info(f"Step {global_step}: Syncing actor_train state to guard (in-memory backup)...")

        # 在每个 worker 内部，调用 worker 的 sync_to_guard 方法
        # 这会在同一个 worker 内部复制 model 和 optimizer 的状态到 guard 对象
        with Timer(name="sync_to_guard", logger=None) as sync_timer:
            refs = [
                worker.sync_to_guard.remote()
                for worker in self.actor_train.workers
            ]
            ray.get(refs)
        logger.info(f"Step {global_step}: State synced to guard, time={sync_timer.last:.2f}s")

    def _sync_from_guard(self, global_step: int):
        """从 guard 恢复 actor_train 的状态（回退）

        在同一个 worker 内部，直接调用 strategy 的方法从 guard 对象恢复状态。
        避免跨 worker 通信的复杂性。

        Args:
            global_step: 当前 step 编号
        """
        logger.info(f"Step {global_step}: Syncing from guard to actor_train (rollback)...")

        # 在每个 worker 内部，调用 worker 的 sync_from_guard 方法
        # 这会从同一个 worker 内部的 guard 对象恢复 model 和 optimizer 的状态
        with Timer(name="sync_from_guard", logger=None) as sync_timer:
            refs = [
                worker.sync_from_guard.remote()
                for worker in self.actor_train.workers
            ]
            ray.get(refs)
        logger.info(f"Step {global_step}: State synced from guard, time={sync_timer.last:.2f}s")

    def _collect_update_norm_metrics(self, global_step: int) -> Dict[str, float]:
        """聚合 actor_train workers 的参数更新范数。"""
        refs = [
            worker.compute_update_norm_from_guard.remote()
            for worker in self.actor_train.workers
        ]
        shard_norms = ray.get(refs)
        shard_norms = [v for v in shard_norms if v is not None]
        if not shard_norms:
            logger.warning(f"Step {global_step}: update_norm is not available on any worker")
            return {}
        shard_norms_np = np.array(shard_norms, dtype=np.float64)
        return {
            "validation/update_norm_shard_l2_mean": float(shard_norms_np.mean()),
            "validation/update_norm_shard_l2_max": float(shard_norms_np.max()),
            "validation/update_norm_shard_l2_min": float(shard_norms_np.min()),
        }

    def _set_actor_learning_rate(
        self,
        global_step: int,
        learning_rate: float,
        metric_prefix: str = "validation/lr_adaptation",
    ) -> Dict[str, float]:
        """Set actor_train learning rate on all workers."""
        refs = [
            worker.set_learning_rate.remote(learning_rate)
            for worker in self.actor_train.workers
        ]
        worker_results = ray.get(refs)
        old_lrs = [
            result.get("old_lr")
            for result in worker_results
            if result is not None and result.get("old_lr") is not None
        ]
        new_lrs = [
            result.get("new_lr")
            for result in worker_results
            if result is not None and result.get("new_lr") is not None
        ]
        logger.info(
            f"Step {global_step}: actor learning rate updated "
            f"from {old_lrs[:4]} to {learning_rate}"
        )
        metrics = {f"{metric_prefix}/actor_lr": learning_rate}
        if old_lrs:
            metrics[f"{metric_prefix}/actor_lr_old_mean"] = float(np.mean(old_lrs))
        if new_lrs:
            metrics[f"{metric_prefix}/actor_lr_worker_mean"] = float(np.mean(new_lrs))
        if metric_prefix == "validation/lr_adaptation":
            metrics["validation/lr_adaptation_actor_lr"] = learning_rate
            if old_lrs:
                metrics["validation/lr_adaptation_actor_lr_old_mean"] = float(np.mean(old_lrs))
            if new_lrs:
                metrics["validation/lr_adaptation_actor_lr_worker_mean"] = float(np.mean(new_lrs))
        return metrics

    def _maybe_apply_baseline_lr_decay(self, global_step: int) -> Dict[str, float]:
        """Apply baseline LR decay independent of update validation."""
        metrics = {
            "baseline_lr_decay/enabled": 1.0 if self.lr_decay_for_baseline else 0.0,
            "baseline_lr_decay/current_lr": self.baseline_current_actor_lr,
        }
        if not self.lr_decay_for_baseline:
            return metrics

        step_index = global_step + 1
        metrics["baseline_lr_decay/step_index"] = float(step_index)
        metrics["baseline_lr_decay/action"] = 0.0
        if step_index % self.lr_decay_for_baseline_decay_steps != 0:
            return metrics

        new_lr = max(self.baseline_current_actor_lr / 2.0, self.lr_decay_for_baseline_min_lr)
        if np.isclose(new_lr, self.baseline_current_actor_lr):
            return metrics

        metrics.update(self._set_actor_learning_rate(global_step, new_lr, metric_prefix="baseline_lr_decay"))
        self.baseline_current_actor_lr = new_lr
        metrics["baseline_lr_decay/current_lr"] = new_lr
        metrics["baseline_lr_decay/action"] = -1.0
        metrics["baseline_lr_decay/decayed_lr"] = new_lr
        return metrics

    def _maybe_adapt_actor_lr(self, global_step: int, rollback_rate: float) -> Dict[str, float]:
        """Adapt actor LR from recent update-validation rollback rate."""
        metrics = {
            "validation/lr_adaptation_enabled": 1.0 if self.uv_lr_adaptation_enable else 0.0,
            "validation/lr_adaptation_current_lr": self.uv_current_actor_lr,
        }
        if not self.uv_lr_adaptation_enable:
            return metrics
        if len(self.uv_rollback_history) < self.uv_lr_adaptation_window:
            return metrics
        if (
            self.uv_lr_adaptation_last_step >= 0
            and global_step - self.uv_lr_adaptation_last_step < self.uv_lr_adaptation_interval
        ):
            return metrics

        new_lr = self.uv_current_actor_lr
        action = "keep"
        if rollback_rate > self.uv_lr_adaptation_high_rollback_rate:
            new_lr = self.uv_current_actor_lr * self.uv_lr_adaptation_decrease_factor
            action = "decrease"
        elif rollback_rate < self.uv_lr_adaptation_low_rollback_rate:
            new_lr = self.uv_current_actor_lr * self.uv_lr_adaptation_increase_factor
            action = "increase"

        if self.uv_lr_adaptation_min_lr is not None:
            new_lr = max(new_lr, self.uv_lr_adaptation_min_lr)
        if self.uv_lr_adaptation_max_lr is not None:
            new_lr = min(new_lr, self.uv_lr_adaptation_max_lr)

        metrics["validation/lr_adaptation_action"] = {
            "keep": 0.0,
            "decrease": -1.0,
            "increase": 1.0,
        }[action]
        metrics["validation/lr_adaptation_rollback_rate"] = rollback_rate

        if action == "keep" or np.isclose(new_lr, self.uv_current_actor_lr):
            return metrics

        metrics.update(self._set_actor_learning_rate(global_step, new_lr))
        self.uv_current_actor_lr = new_lr
        self.uv_lr_adaptation_last_step = global_step
        return metrics

    def _capture_rollback_debug_snapshot(self, global_step: int, tag: str):
        """在 actor_train workers 上保存一份 rollback 调试快照。"""
        # # logger.error(f"[UV] step={global_step} ROLLBACK_DEBUG BEFORE capture tag={tag}")
        refs = [
            worker.capture_debug_state_snapshot.remote(tag)
            for worker in self.actor_train.workers
        ]
        ray.get(refs)
        # # logger.error(f"[UV] step={global_step} ROLLBACK_DEBUG AFTER capture tag={tag}")

    def _compare_rollback_debug_snapshot(self, global_step: int, tag: str):
        """在 actor_train workers 上比较当前状态与 rollback 调试快照。"""
        # # logger.error(f"[UV] step={global_step} ROLLBACK_DEBUG BEFORE compare tag={tag}")
        refs = [
            worker.compare_debug_state_snapshot.remote(tag)
            for worker in self.actor_train.workers
        ]
        ray.get(refs)
        # # logger.error(f"[UV] step={global_step} ROLLBACK_DEBUG AFTER compare tag={tag}")

    def _prepare_validation_prompts(self, samples: Optional[List[Dict]] = None) -> DataProto:
        """准备验证用的 prompts DataProto。
        samples: 为 None 时使用 self.uv_validation_samples（data_mode=fixed）；否则使用传入的样本列表（data_mode=random_per_step）。
        """
        if samples is None:
            samples = self.uv_validation_samples
        # 收集所有验证样本的数据
        input_ids_list = []
        attention_mask_list = []
        domain_list = []
        non_tensor_data = {key: [] for key in samples[0].keys()
                          if key not in ["input_ids", "attention_mask", "domain"]}

        for sample in samples:
            input_ids = sample["input_ids"]
            if isinstance(input_ids, list):
                input_ids = torch.tensor(input_ids)
            input_ids_list.append(input_ids)

            if "attention_mask" in sample:
                attention_mask = sample["attention_mask"]
                if isinstance(attention_mask, list):
                    attention_mask = torch.tensor(attention_mask)
            else:
                attention_mask = torch.ones_like(input_ids)
            attention_mask_list.append(attention_mask)

            domain_list.append(sample.get("domain", "math_rule"))

            for key in non_tensor_data.keys():
                # 采样时已确保 required_fields 存在，其他字段用 get 获取
                non_tensor_data[key].append(sample.get(key))

        # Pad sequences to same length
        # 参考正常训练流程：collect_fn 将所有 prompts pad 到 prompt_length（配置值）
        # 使用 pipeline_config.prompt_length 而不是 max_len，与正常训练一致
        target_prompt_length = self.pipeline_config.prompt_length
        pad_token_id = self.tokenizer.pad_token_id

        padded_input_ids = []
        padded_attention_mask = []
        for input_ids, attention_mask in zip(input_ids_list, attention_mask_list):
            padding_length = target_prompt_length - len(input_ids)
            if padding_length > 0:
                # Left padding (与正常训练一致)
                input_ids = torch.cat([
                    torch.full((padding_length,), pad_token_id, dtype=input_ids.dtype),
                    input_ids
                ])
                attention_mask = torch.cat([
                    torch.zeros(padding_length, dtype=attention_mask.dtype),
                    attention_mask
                ])
            elif padding_length < 0:
                # 如果超过 prompt_length，截断
                input_ids = input_ids[:target_prompt_length]
                attention_mask = attention_mask[:target_prompt_length]
            padded_input_ids.append(input_ids)
            padded_attention_mask.append(attention_mask)

        input_ids_tensor = torch.stack(padded_input_ids)
        attention_mask_tensor = torch.stack(padded_attention_mask)
        position_ids_tensor = torch.clamp(torch.cumsum(attention_mask_tensor, dim=-1) - 1, min=0)

        from roll.distributed.scheduler.protocol import TensorDict
        batch = TensorDict(
            {
                "input_ids": input_ids_tensor,
                "attention_mask": attention_mask_tensor,
                "position_ids": position_ids_tensor,
                "prompts": input_ids_tensor.clone(),
            },
            batch_size=len(samples),
        )

        # 添加 domain 信息
        batch["domain"] = torch.tensor([hash(d) % (2**31) for d in domain_list])  # 简单编码

        # 构建 non_tensor_batch
        non_tensor_batch = {"domain": np.array(domain_list, dtype=object)}
        for key, values in non_tensor_data.items():
            # 确保所有样本的字段都被包含，即使有些是 None
            # 长度检查是为了验证数据完整性
            if len(values) == len(samples):
                non_tensor_batch[key] = make_non_tensor_array(values)
            else:
                logger.warning(f"Update Validation: field '{key}' has {len(values)} values but expected {len(samples)}, skipping")

        return DataProto(
            batch=batch,
            non_tensor_batch=non_tensor_batch,
            meta_info={}
        )

    def _generate_validation_requests_legacy(
        self,
        single_prompts: List[DataProto],
    ) -> Dict[int, DataProto]:
        """Run the original worker-local blocking request path."""
        batch_size = len(single_prompts)
        num_workers = len(self.actor_infer.workers)
        max_inflight_requests = max(1, min(self.uv_generate_concurrency, batch_size))
        index_to_response = {}
        for start in range(0, batch_size, max_inflight_requests):
            refs_with_index: List[Tuple[int, ray.ObjectRef]] = []
            end = min(start + max_inflight_requests, batch_size)
            for i in range(start, end):
                worker_id = i % num_workers
                ref = self.actor_infer.workers[worker_id].generate_request.remote(single_prompts[i])
                refs_with_index.append((i, ref))

            response_list = ray.get([ref for _, ref in refs_with_index])
            index_to_response.update(
                {idx: response for (idx, _), response in zip(refs_with_index, response_list)}
            )
        return index_to_response

    def _generate_validation_requests_sglang_sync(
        self,
        single_prompts: List[DataProto],
    ) -> Dict[int, DataProto]:
        """Use the proven synchronous SGLang generation path for UV metrics."""
        index_to_response = {}
        num_workers = len(self.actor_infer.workers)
        for start in range(0, len(single_prompts), num_workers):
            indexed_prompts = list(enumerate(single_prompts[start : start + num_workers], start=start))
            responses = ray.get(
                [
                    self.actor_infer.workers[index % num_workers].generate_request_sync.remote(prompt)
                    for index, prompt in indexed_prompts
                ]
            )
            index_to_response.update(dict(zip((index for index, _ in indexed_prompts), responses)))
            logger.info(
                f"SGLang UV synchronous generation completed "
                f"{min(start + num_workers, len(single_prompts))}/{len(single_prompts)} prompts"
            )
        return index_to_response

    def _generate_validation_requests_scheduler(
        self,
        single_prompts: List[DataProto],
    ) -> Dict[int, DataProto]:
        """Keep a sliding request window full through RequestScheduler."""
        if self.uv_request_scheduler is None:
            raise RuntimeError("Update Validation request scheduler is not initialized")

        batch_size = len(single_prompts)
        max_inflight_requests = max(1, min(self.uv_generate_concurrency, batch_size))
        pending: Dict[ray.ObjectRef, int] = {}
        index_to_response = {}
        next_index = 0

        def submit(index: int) -> None:
            prompt = single_prompts[index]
            prompt.meta_info["src_rank"] = index
            prompt.meta_info["response_callback_fn"] = self.uv_request_scheduler.report_response.remote
            ref = self.uv_request_scheduler.generate_raw_request.remote(data=prompt)
            pending[ref] = index

        while next_index < max_inflight_requests:
            submit(next_index)
            next_index += 1

        while pending:
            ready_refs, _ = ray.wait(list(pending), num_returns=1)
            for ref in ready_refs:
                index = pending.pop(ref)
                index_to_response[index] = ray.get(ref)
                if next_index < batch_size:
                    submit(next_index)
                    next_index += 1

        return index_to_response

    def _generate_validation_batch(self, global_step: int, rollout_batch: Optional[DataProto] = None) -> DataProto:
        """生成验证 batch

        使用当前的 actor_infer（已同步新参数）进行推理
        根据配置使用 legacy generate_request 或 RequestScheduler 获取 infer_logprobs

        Args:
            global_step: 当前 step 编号
            rollout_batch: 当前 step 的 rollout batch，仅 data_mode=\"rollout_batch\" 时使用

        Returns:
            包含推理结果和 reward 的 DataProto
        """
        # 1. 按 data_mode 准备 validation prompts
        if self.uv_data_mode == "fixed":
            validation_prompts = self._prepare_validation_prompts()
        elif self.uv_data_mode == "rollout_batch":
            if rollout_batch is None:
                raise ValueError("Update Validation data_mode=rollout_batch requires rollout_batch to be passed to _generate_validation_batch")
            validation_prompts = self._prepare_validation_prompts_from_rollout_batch(rollout_batch)
        else:  # random_per_step
            samples = self._sample_validation_from_dataset()
            if not samples:
                raise RuntimeError("Update Validation data_mode=random_per_step sampled 0 samples")
            validation_prompts = self._prepare_validation_prompts(samples=samples)

        # 2. 设置生成配置
        validation_generating_args = copy.deepcopy(self.pipeline_config.actor_infer.generating_args)
        validation_generating_args.num_return_sequences = self.uv_num_return_sequences

        validation_generation_config = validation_generating_args.to_dict()
        # 强制设置 logprobs=1，确保生成时返回 logprobs
        validation_generation_config["logprobs"] = 1

        validation_prompts.meta_info["generation_config"] = validation_generation_config
        validation_prompts.meta_info["global_step"] = global_step
        validation_prompts.meta_info["is_offload_states"] = False

        # Log validation prompts non_tensor_batch keys for debugging
        logger.info(f"Update Validation: validation_prompts.non_tensor_batch keys = {list(validation_prompts.non_tensor_batch.keys()) if validation_prompts.non_tensor_batch else None}")
        if validation_prompts.non_tensor_batch and "ground_truth" in validation_prompts.non_tensor_batch:
            logger.info(f"Update Validation: ground_truth sample = {validation_prompts.non_tensor_batch['ground_truth'][:3] if len(validation_prompts.non_tensor_batch['ground_truth']) >= 3 else validation_prompts.non_tensor_batch['ground_truth']}")

        # 3. 获取原始 callback 数据中的 output_logprobs，再统一生成 infer_logprobs。
        # 注意：不能复用本 step 的 step_generate，因为那是在 train 之前用旧策略生成的；
        # update_validation 必须在 train 之后用新策略（已同步到 actor_infer）对固定验证集再生成一次。
        batch_size = validation_prompts.batch["input_ids"].shape[0]

        is_sglang = self.pipeline_config.actor_infer.strategy_args.strategy_name == "sglang"
        if not is_sglang:
            self.actor_infer.start_server(
                data=DataProto(meta_info={"global_step": global_step, "is_offload_states": False})
            )

        # 3.1 为每个 prompt 创建一个 inference 请求。
        single_prompts: List[DataProto] = []
        for i in range(batch_size):
            single_prompt_batch = {
                "input_ids": validation_prompts.batch["input_ids"][i:i+1],
                "attention_mask": validation_prompts.batch["attention_mask"][i:i+1],
            }
            if "position_ids" in validation_prompts.batch:
                single_prompt_batch["position_ids"] = validation_prompts.batch["position_ids"][i:i+1]
            if "prompts" in validation_prompts.batch:
                single_prompt_batch["prompts"] = validation_prompts.batch["prompts"][i:i+1]

            single_prompt = DataProto(
                batch=TensorDict(single_prompt_batch, batch_size=1),
                meta_info=validation_prompts.meta_info.copy()
            )
            single_prompt.meta_info["request_id"] = f"validation_{global_step}_{i}"
            if validation_prompts.non_tensor_batch is not None:
                single_prompt.non_tensor_batch = {
                    k: repeat_non_tensor_value(v[i], 1)
                    for k, v in validation_prompts.non_tensor_batch.items()
                }
            single_prompts.append(single_prompt)

        try:
            if is_sglang:
                index_to_response = self._generate_validation_requests_sglang_sync(single_prompts)
            elif self.uv_generate_mode == "scheduler":
                index_to_response = self._generate_validation_requests_scheduler(single_prompts)
            else:
                index_to_response = self._generate_validation_requests_legacy(single_prompts)
        except Exception:
            if self.uv_request_scheduler is not None:
                ray.get(self.uv_request_scheduler.abort_request.remote())
            raise
        finally:
            if not is_sglang:
                self.actor_infer.stop_server()

        # 3.3 按 index 顺序处理每个 response，整理为 raw_outputs_data
        raw_outputs_data = []
        for i in range(batch_size):
            response_data = index_to_response.get(i)
            if response_data is None:
                continue
            if not is_report_data_finished(response_data):
                logger.warning(f"Update Validation: Request {i} failed, skipping")
                continue

            single_prompt = single_prompts[i]
            output_token_ids = response_data.meta_info.get("output_token_ids", [])
            output_logprobs = response_data.meta_info.get("output_logprobs", None)

            eos_token_id = response_data.meta_info.get("eos_token_id")
            pad_token_id = response_data.meta_info.get("pad_token_id")
            if eos_token_id is None or pad_token_id is None:
                gen_config = response_data.meta_info.get("generation_config", validation_generation_config)
                if eos_token_id is None:
                    eos_token_id = gen_config.get("eos_token_id")
                    if isinstance(eos_token_id, list):
                        eos_token_id = eos_token_id[0]
                if pad_token_id is None:
                    pad_token_id = gen_config.get("pad_token_id")
            if eos_token_id is None or pad_token_id is None:
                if hasattr(self, 'tokenizer') and self.tokenizer is not None:
                    if eos_token_id is None:
                        eos_token_id = self.tokenizer.eos_token_id
                    if pad_token_id is None:
                        pad_token_id = self.tokenizer.pad_token_id

            output_tokens = [torch.tensor(token_ids) for token_ids in output_token_ids]
            output_tensor = pad_sequence(output_tokens, batch_first=True, padding_value=pad_token_id)
            if hasattr(single_prompt.batch["input_ids"], 'device'):
                output_tensor = output_tensor.to(single_prompt.batch["input_ids"].device)
            output_tensor = concatenate_input_and_output(
                input_ids=single_prompt.batch["input_ids"],
                output_ids=output_tensor,
                num_return_sequences=len(output_tokens)
            )

            raw_outputs_data.append({
                "single_prompt": single_prompt,
                "output_tensor": output_tensor,
                "output_tokens": output_tokens,
                "output_logprobs": output_logprobs,
                "eos_token_id": eos_token_id,
                "pad_token_id": pad_token_id,
                "index": i,
            })

        # 3.0.1 确定统一的 sequence_length（使用所有 outputs 中的最大值）
        # 参考正常训练流程：整个 batch 一起处理时，pad_sequence 会将所有 sequences pad 到最大长度
        if raw_outputs_data:
            max_sequence_length = max(
                data["output_tensor"].shape[-1]
                for data in raw_outputs_data
            )
            logger.info(
                f"Update Validation: Using unified sequence_length={max_sequence_length} "
                f"for all {len(raw_outputs_data)} outputs (max of actual lengths)"
            )
        else:
            raise RuntimeError("Update Validation: No valid outputs collected")

        # 3.7 使用统一的 sequence_length 处理所有 outputs
        all_outputs = []
        for data in raw_outputs_data:
            i = data["index"]
            single_prompt = data["single_prompt"]
            output_tensor = data["output_tensor"]
            output_tokens = data["output_tokens"]
            output_logprobs = data["output_logprobs"]
            if output_logprobs is not None and len(output_logprobs) != len(output_tokens):
                logger.warning(
                    "Update Validation: output_logprobs count does not match generated sequences "
                    f"({len(output_logprobs)} != {len(output_tokens)}); dropping logprobs for request {i}"
                )
                output_logprobs = None
            eos_token_id = data["eos_token_id"]
            pad_token_id = data["pad_token_id"]

            # 记录关键信息用于调试
            prompt_length = single_prompt.batch["input_ids"].shape[1]
            actual_sequence_length = output_tensor.shape[-1]
            # # logger.error(
            #     f"Update Validation: Output {i} BEFORE postprocess_generate - "
            #     f"prompt_length={prompt_length}, "
            #     f"actual_sequence_length={actual_sequence_length}, "
            #     f"unified_sequence_length={max_sequence_length}, "
            #     f"output_tensor.shape={output_tensor.shape}, "
            #     f"expected_response_length={max_sequence_length - prompt_length}"
            # )

            # 使用统一的 sequence_length 调用 postprocess_generate
            # 参考 generate_scheduler.py:1356，单个请求使用 output_tensor.shape[-1]
            # 但这里我们需要统一长度，所以使用 max_sequence_length
            output = postprocess_generate(
                prompts=single_prompt,
                output=output_tensor,
                num_return_sequences=len(output_tokens),
                sequence_length=max_sequence_length,  # ✅ 使用统一的最大长度
                eos_token_id=eos_token_id,
                pad_token_id=pad_token_id,
                pad_to_seq_len=True,
                output_logprobs=output_logprobs,
            )

            # 记录 postprocess_generate 后的关键信息
            if output.batch is not None:
                responses_shape = output.batch["responses"].shape
                actual_response_length = responses_shape[1]  # 这是 responses tensor 的实际第二维长度
                # # logger.error(
                #     f"Update Validation: Output {i} AFTER postprocess_generate - "
                #     f"responses.shape={responses_shape}, "
                #     f"actual_response_tensor_length={actual_response_length}, "
                #     f"prompt_length={prompt_length}, "
                #     f"total_sequence_length={responses_shape[1] + prompt_length}"
                # )

            # 3.8 扩展 non_tensor_batch（每个 prompt 对应 num_return_sequences 个序列）
            if single_prompt.non_tensor_batch is not None:
                output.non_tensor_batch = {}
                for k, v in single_prompt.non_tensor_batch.items():
                    # 每个 prompt 对应 len(output_tokens) 个序列
                    output.non_tensor_batch[k] = repeat_non_tensor_value(v[0], len(output_tokens))

            # 记录每个 output 的 batch 形状信息用于调试
            if output.batch is not None:
                batch_shapes = {}
                for key, value in output.batch.items():
                    if isinstance(value, torch.Tensor):
                        batch_shapes[key] = list(value.shape)
                    else:
                        batch_shapes[key] = f"non-tensor: {type(value)}"
                # # logger.error(f"Update Validation: Output {i} batch shapes: {batch_shapes}")
                # # logger.error(f"Update Validation: Output {i} batch_size: {output.batch.batch_size}")
                # # logger.error(f"Update Validation: Output {i} output_tensor shape: {output_tensor.shape}")
                # # logger.error(f"Update Validation: Output {i} num_return_sequences: {len(output_tokens)}")
            else:
                logger.error(f"Update Validation: Output {i} batch is None!")

            all_outputs.append(output)

        # 4. 合并所有输出
        if all_outputs:
            # 在 concat 之前记录所有 outputs 的详细信息
            # logger.error(f"Update Validation: About to concat {len(all_outputs)} outputs")

            # 关键：检查所有 outputs 的 responses 形状是否一致
            responses_shapes = []
            for idx, output in enumerate(all_outputs):
                if output.batch is not None:
                    batch_info = {}
                    for key, value in output.batch.items():
                        if isinstance(value, torch.Tensor):
                            batch_info[key] = {
                                "shape": list(value.shape),
                                "dtype": str(value.dtype),
                                "device": str(value.device)
                            }
                            # 特别记录 responses 的形状
                            if key == "responses":
                                responses_shapes.append(list(value.shape))
                        else:
                            batch_info[key] = f"non-tensor: {type(value)}"
                    # # logger.error(f"Update Validation: Output {idx} before concat - batch_info: {batch_info}")
                    # # logger.error(f"Update Validation: Output {idx} before concat - batch_size: {output.batch.batch_size}")
                else:
                    logger.error(f"Update Validation: Output {idx} before concat - batch is None!")

            # 检查 responses 形状是否一致
            if responses_shapes:
                unique_shapes = set(tuple(shape) for shape in responses_shapes)
                # # logger.error(
                #     f"Update Validation: responses shapes summary - "
                #     f"total_outputs={len(responses_shapes)}, "
                #     f"unique_shapes={unique_shapes}, "
                #     f"all_shapes={responses_shapes}"
                # )
                if len(unique_shapes) > 1:
                    logger.error(
                        f"Update Validation: [ERROR] responses shapes are NOT consistent! "
                        f"This will cause concat to fail. "
                        f"Unique shapes: {unique_shapes}",
                    )

            generate_output = DataProto.concat(all_outputs)
            has_logprobs = "infer_logprobs" in generate_output.batch
            logger.info(f"Update Validation: Generated batch with infer_logprobs={has_logprobs}, "
                       f"shape={generate_output.batch.get('infer_logprobs', None).shape if has_logprobs else None}")
            if not has_logprobs:
                raise RuntimeError(
                    "Update Validation requires infer_logprobs from actor_infer.generate_request; "
                    "ensure validation generation sets generation_config.logprobs=1 and the selected "
                    "inference strategy exposes per-token logprobs."
                )
        else:
            # 如果所有请求都失败，抛出异常
            raise RuntimeError("Update Validation: All generate_request failed, cannot generate validation batch")

        # 5. 计算 rewards（使用第一个 reward domain）
        domain = list(self.rewards.keys())[0]
        reward_cluster = self.rewards[domain]

        # 添加必要的字段
        output_count = generate_output.batch.batch_size[0]
        if generate_output.non_tensor_batch is None:
            generate_output.non_tensor_batch = {}
        generate_output.non_tensor_batch["rollout_id"] = np.array(
            [str(uuid.uuid4()) for _ in range(output_count)], dtype=object
        )

        reward_refs = reward_cluster.compute_rewards(generate_output, blocking=False)
        reward_output = DataProto.materialize_concat(data_refs=reward_refs)

        # 合并 reward 结果
        if "scores" in reward_output.batch:
            generate_output.batch["scores"] = reward_output.batch["scores"]
        if "response_level_rewards" in reward_output.batch:
            generate_output.batch["response_level_rewards"] = reward_output.batch["response_level_rewards"]
        elif "scores" in reward_output.batch:
            generate_output.batch["response_level_rewards"] = reward_output.batch["scores"].float()

        if "response_mask" not in generate_output.batch:
            attention_mask = generate_output.batch["attention_mask"]
            prompt_length = validation_prompts.batch["prompts"].shape[1]
            response_mask = torch.zeros_like(attention_mask)
            response_mask[:, prompt_length:] = attention_mask[:, prompt_length:]
            generate_output.batch["response_mask"] = response_mask

        if "response_level_rewards" not in generate_output.batch:
            scores = generate_output.batch.get("scores", torch.zeros(generate_output.batch.batch_size[0]))
            generate_output.batch["response_level_rewards"] = scores.float()

        return generate_output

    def _compute_validation_advantage(self, validation_batch: DataProto) -> tuple[float, dict]:
        """计算验证 batch 的优势值

        Follow 正常训练流程中的 reward normalization:
        1. reward_postprocess: reward normalization (group norm for GRPO)
        2. expand_to_token_level: 展开到 token level
        3. compute_reinforce_return: 计算优势值

        Args:
            validation_batch: 包含推理结果和 reward 的 DataProto

        Returns:
            avg_advantage: 聚合后的优势值（标量）
            metrics: 相关指标
        """
        metrics = {}


        if "old_log_probs" not in validation_batch.batch:
            logger.info("Update Validation: Computing old_log_probs using guard model (old policy)...")
            validation_batch.meta_info["is_offload_states"] = False
            old_log_probs_refs = self.actor_train.compute_log_probs(validation_batch, blocking=False)
            old_log_probs = DataProto.materialize_concat(data_refs=old_log_probs_refs)
            validation_batch.batch["old_log_probs"] = old_log_probs.batch["log_probs"]
            logger.info("Update Validation: old_log_probs computed successfully")

        # 1. reward_postprocess
        domain_key = list(self.running.keys())[0]
        running_ctrl = self.running[domain_key]

        # 执行 reward normalization（和正常训练流程一致）
        validation_batch, reward_metrics = reward_postprocess(
            validation_batch,
            self.pipeline_config,
            running_ctrl
        )
        metrics.update({f"validation/{k}": v for k, v in reward_metrics.items()})

        # 2. Expand to Token Level
        token_level_rewards = expand_to_token_level(validation_batch)
        validation_batch.batch["token_level_rewards"] = token_level_rewards

        # 3. Compute Advantages
        response_mask = validation_batch.batch["response_mask"][:, 1:]
        token_level_rewards = validation_batch.batch["token_level_rewards"]

        # 截取到与 response_mask 相同长度
        if token_level_rewards.shape[1] > response_mask.shape[1]:
            token_level_rewards = token_level_rewards[:, :response_mask.shape[1]]
        elif token_level_rewards.shape[1] < response_mask.shape[1]:
            padding = torch.zeros(
                token_level_rewards.shape[0],
                response_mask.shape[1] - token_level_rewards.shape[1],
                device=token_level_rewards.device,
                dtype=token_level_rewards.dtype
            )
            token_level_rewards = torch.cat([token_level_rewards, padding], dim=1)

        # 计算 advantages（参考 compute_reinforce_return）
        advantages, returns = compute_reinforce_return(
            token_level_rewards=token_level_rewards.float(),
            gamma=self.pipeline_config.gamma,
            lambd=self.pipeline_config.lambd
        )

        old_lp = validation_batch.batch.get("old_log_probs")
        infer_lp = validation_batch.batch.get("infer_logprobs")
        # Calculate seq-level IS ratio
        log_ratio_seq = masked_sum(old_lp - infer_lp, response_mask, dim=-1)
        # Clamp log ratio to prevent extreme importance sampling ratios
        log_ratio_seq_geo = masked_mean(old_lp - infer_lp, response_mask, dim=-1)
        _importance_ratio_seq = log_ratio_seq.exp()
        importance_ratio_seq = _importance_ratio_seq.clamp(max=1.0)
        importance_ratio_seq_geo = log_ratio_seq_geo.exp()

        # Response-level advantage: one scalar per sequence (masked mean over tokens)
        response_level_advantages = masked_mean(advantages, response_mask, dim=-1)
        # Seq-level IS advantages: response-level adv * response-level IS ratio
        seq_level_is_advantages = response_level_advantages * importance_ratio_seq
        seq_level_is_advantages_geo = response_level_advantages * importance_ratio_seq_geo

        # Record metrics: mean over responses (not token-level mean)
        seq_level_is_adv_mean = seq_level_is_advantages.mean().item()
        seq_level_is_adv_mean_geo = seq_level_is_advantages_geo.mean().item()
        # 记录其他指标
        scores = validation_batch.batch.get("scores", None)
        response_level_rewards = validation_batch.batch.get("response_level_rewards", None)

        if scores is not None:
            metrics["validation/scores_mean"] = scores.float().mean().item()
            metrics["validation/scores_std"] = scores.float().std().item()

        if response_level_rewards is not None:
            metrics["validation/normalized_reward_mean"] = response_level_rewards.float().mean().item()
            metrics["validation/normalized_reward_std"] = response_level_rewards.float().std().item()

        metrics["validation/token_rewards_sum_mean"] = token_level_rewards.sum(dim=-1).mean().item()
        metrics["validation/advantages_mean"] = seq_level_is_adv_mean
        metrics["validation/advantages_mean_geo"] = seq_level_is_adv_mean_geo

        # 分布统计：std / max / min / 离群点比例（|x - mean| > 3*std）
        def _dist_stats(t: torch.Tensor, prefix: str):
            t = t.float().detach()
            if t.numel() == 0:
                return
            mean = t.mean()
            std = t.std(unbiased=False)
            metrics[f"{prefix}_std"] = std.item()
            metrics[f"{prefix}_max"] = t.max().item()
            metrics[f"{prefix}_min"] = t.min().item()
            if t.numel() > 1 and std.item() > 0:
                outlier_frac = ((t - mean).abs() > 3.0 * std).float().mean().item()
            else:
                outlier_frac = 0.0
            metrics[f"{prefix}_outlier_frac"] = outlier_frac

        _dist_stats(seq_level_is_advantages, "validation/advantages")
        _dist_stats(seq_level_is_advantages_geo, "validation/advantages_geo")

        # IS ratio 分布统计（mean / std / max / min / 离群点比例）
        _dist_stats(_importance_ratio_seq, "validation/is_ratio_seq")
        metrics["validation/is_ratio_seq_mean"] = _importance_ratio_seq.float().mean().item()
        _dist_stats(importance_ratio_seq, "validation/is_ratio_seq_clamped")
        metrics["validation/is_ratio_seq_clamped_mean"] = importance_ratio_seq.float().mean().item()
        _dist_stats(importance_ratio_seq_geo, "validation/is_ratio_seq_geo")
        metrics["validation/is_ratio_seq_geo_mean"] = importance_ratio_seq_geo.float().mean().item()

        # 返回 seq_level_is_adv_mean 作为 avg_advantage（用于阈值判断）
        avg_advantage = seq_level_is_adv_mean_geo
        return avg_advantage, metrics

    def _prepare_offpolicy_rollout_cache(self, rollout_batch: DataProto) -> DataProto:
        """为 off-policy 多步估计缓存必要字段。"""
        required_batch_keys = ["input_ids", "attention_mask", "response_mask", "infer_logprobs"]
        optional_batch_keys = ["position_ids", "advantages", "token_level_rewards"]
        missing = [k for k in required_batch_keys if k not in rollout_batch.batch]
        if missing:
            raise ValueError(f"Update Validation off-policy rollout cache missing required keys: {missing}")

        batch_keys = required_batch_keys + [k for k in optional_batch_keys if k in rollout_batch.batch]
        non_tensor_keys = []
        if rollout_batch.non_tensor_batch and "multi_modal_data" in rollout_batch.non_tensor_batch:
            non_tensor_keys.append("multi_modal_data")
        cached = rollout_batch.select(
            batch_keys=batch_keys,
            non_tensor_batch_keys=non_tensor_keys if non_tensor_keys else None,
            meta_info_keys=["global_step"] if rollout_batch.meta_info and "global_step" in rollout_batch.meta_info else None,
            deepcopy=True,
        ).clone()
        cached.to("cpu")
        return cached

    def _push_offpolicy_rollout_history(self, global_step: int, rollout_batch: DataProto):
        cached = self._prepare_offpolicy_rollout_cache(rollout_batch)
        self.uv_offpolicy_rollout_history.append((global_step, cached))

    def _collect_offpolicy_rollouts_window(self, global_step: int, rollout_batch: Optional[DataProto] = None) -> tuple[DataProto, dict]:
        """
        收集 off-policy 最近 n 步窗口 [t-n+1, t] 的 rollouts，并拼接成一个 validation_batch。
        """
        metrics = {}
        start_step = global_step - self.uv_offpolicy_rollout_n + 1
        window = []
        for step_i, cached in self.uv_offpolicy_rollout_history:
            if start_step <= step_i <= global_step:
                window.append((step_i, cached))

        if not window and rollout_batch is not None:
            logger.warning("Update Validation(off-policy): rollout history empty, use current step rollout only.")
            window = [(global_step, self._prepare_offpolicy_rollout_cache(rollout_batch))]

        if not window:
            raise RuntimeError("Update Validation(off-policy): no rollout data available in requested window.")

        # 按 step 排序，保证拼接顺序稳定
        window = sorted(window, key=lambda x: x[0])
        batches = [cached.clone() for _, cached in window]
        validation_batch = DataProto.concat(batches) if len(batches) > 1 else batches[0]
        validation_batch.meta_info["is_offload_states"] = False

        metrics["validation/offpolicy/window_start_step"] = float(window[0][0])
        metrics["validation/offpolicy/window_end_step"] = float(window[-1][0])
        metrics["validation/offpolicy/window_steps_used"] = float(len(window))
        metrics["validation/offpolicy/window_samples_used"] = float(validation_batch.batch.batch_size[0])
        return validation_batch, metrics

    def _prepare_offpolicy_validation_batch(
        self, global_step: int, rollout_batch: Optional[DataProto] = None
    ) -> tuple[DataProto, dict]:
        """off-policy 模式：复用多步训练轨迹并用更新后 inference 估计 log_probs。"""
        metrics = {}
        validation_batch, window_metrics = self._collect_offpolicy_rollouts_window(
            global_step=global_step,
            rollout_batch=rollout_batch,
        )
        metrics.update(window_metrics)

        with Timer(name="validation_compute_new_infer_log_probs", logger=None) as timer:
            new_infer_log_probs_refs = self.actor_infer.compute_log_probs(validation_batch, blocking=False)
            new_infer_log_probs = DataProto.materialize_concat(data_refs=new_infer_log_probs_refs)
        metrics["time/validation_compute_new_infer_log_probs"] = timer.last
        metrics.update({
            f"validation/{k}": v
            for k, v in reduce_metrics(new_infer_log_probs.meta_info.pop("metrics", {})).items()
        })
        metrics["validation/offpolicy/new_infer_source_actor_infer"] = 1.0

        validation_batch.batch["new_infer_log_probs"] = new_infer_log_probs.batch["log_probs"]
        return validation_batch, metrics

    def _compute_offpolicy_validation_advantage(self, validation_batch: DataProto) -> tuple[float, dict]:
        """off-policy 估计：adv_final = r1 * r2 * adv。"""
        metrics = {}
        response_mask = validation_batch.batch["response_mask"][:, 1:]

        old_infer_lp = validation_batch.batch.get("infer_logprobs")
        new_infer_lp = validation_batch.batch.get("new_infer_log_probs")
        if old_infer_lp is None or new_infer_lp is None:
            raise ValueError(
                "Update Validation off-policy requires infer_logprobs(old inference) "
                "and new_infer_log_probs(new inference)."
            )
        new_trainer_lp = validation_batch.batch.get("new_trainer_log_probs")
        if new_trainer_lp is None:
            validation_batch.meta_info["is_offload_states"] = False
            new_trainer_log_probs_refs = self.actor_train.compute_log_probs(validation_batch, blocking=False)
            new_trainer_log_probs = DataProto.materialize_concat(data_refs=new_trainer_log_probs_refs)
            new_trainer_lp = new_trainer_log_probs.batch["log_probs"]
            validation_batch.batch["new_trainer_log_probs"] = new_trainer_lp
            metrics.update({
                f"validation/{k}": v
                for k, v in reduce_metrics(new_trainer_log_probs.meta_info.pop("metrics", {})).items()
            })

        if "advantages" in validation_batch.batch:
            token_level_advantages = validation_batch.batch["advantages"]
        else:
            token_level_rewards = validation_batch.batch.get("token_level_rewards")
            if token_level_rewards is None:
                token_level_rewards = expand_to_token_level(validation_batch)
            token_level_advantages, _ = compute_reinforce_return(
                token_level_rewards=token_level_rewards.float(),
                gamma=self.pipeline_config.gamma,
                lambd=self.pipeline_config.lambd,
            )

        def _align_to_response_length(tensor: torch.Tensor) -> torch.Tensor:
            if tensor.shape[1] > response_mask.shape[1]:
                return tensor[:, :response_mask.shape[1]]
            if tensor.shape[1] < response_mask.shape[1]:
                padding = torch.zeros(
                    tensor.shape[0],
                    response_mask.shape[1] - tensor.shape[1],
                    device=tensor.device,
                    dtype=tensor.dtype,
                )
                return torch.cat([tensor, padding], dim=1)
            return tensor

        token_level_advantages = _align_to_response_length(token_level_advantages)
        old_infer_lp = _align_to_response_length(old_infer_lp)
        new_infer_lp = _align_to_response_length(new_infer_lp)
        new_trainer_lp = _align_to_response_length(new_trainer_lp)

        response_level_adv = masked_mean(token_level_advantages, response_mask, dim=-1)
        r1 = (masked_mean(new_infer_lp - old_infer_lp, response_mask, dim=-1)).exp()
        r2 = (masked_mean(new_trainer_lp - new_infer_lp, response_mask, dim=-1)).exp()
        final_response_adv = response_level_adv * r1 * r2

        metrics["validation/offpolicy/r1_mean"] = r1.mean().item()
        metrics["validation/offpolicy/r2_mean"] = r2.mean().item()
        metrics["validation/offpolicy/base_adv_mean"] = response_level_adv.mean().item()
        metrics["validation/offpolicy/final_adv_mean"] = final_response_adv.mean().item()
        avg_advantage = final_response_adv.mean().item()
        return avg_advantage, metrics

    def _update_advantage_threshold(self, avg_advantage: float):
        """更新优势值 buffer 和阈值

        Args:
            avg_advantage: 当前 step 的平均优势值
        """
        self.uv_advantage_buffer.append(avg_advantage)

        if len(self.uv_advantage_buffer) >= self.uv_min_steps_for_threshold:
            # 计算分位数作为阈值
            self.uv_advantage_threshold = float(np.percentile(
                list(self.uv_advantage_buffer),
                self.uv_threshold_percentile
            ))

    def _get_acceptance_advantage_threshold(self, global_step: int) -> float:
        """获取当前 step 生效的 acceptance threshold。"""
        if self.uv_acceptance_advantage_threshold_mode == "fixed":
            return float(self.uv_acceptance_advantage_threshold)

        # reverse_decay: 从 a 开始，按线性方式在给定步数内上升到 b
        # threshold(step) = a + (b-a) * min(progress/steps, 1.0)
        progress = max(global_step - self.uv_min_steps_for_threshold, 0)
        start = float(self.uv_acceptance_advantage_threshold_reverse_decay_start)
        end = float(self.uv_acceptance_advantage_threshold_reverse_decay_end)
        total_steps = float(self.uv_acceptance_advantage_threshold_reverse_decay_steps)
        alpha = min(progress / total_steps, 1.0)
        return float(start + (end - start) * alpha)

    def _validate_and_maybe_rollback(self, global_step: int, rollout_batch: Optional[DataProto] = None) -> tuple[bool, dict]:
        """验证模型更新，如果不通过则回退

        Args:
            global_step: 当前 step 编号
            rollout_batch: 当前 step 的 rollout batch，data_mode=\"rollout_batch\" 时用于验证

        Returns:
            accepted: 是否接受此次更新
            metrics: 验证相关的 metrics
        """
        metrics = {}
        # # logger.error(f"[UV] _validate_and_maybe_rollback ENTER step={global_step}")

        # 1. 准备验证 batch（on-policy: 新 rollout；off-policy: 复用训练轨迹）
        if self.uv_est_mode == "off-policy":
            if rollout_batch is None:
                raise ValueError("Update Validation est_mode=off-policy requires rollout_batch")
            metrics["time/validation_generate"] = 0.0
            with Timer(name="validation_prepare_offpolicy", logger=None) as prepare_timer:
                validation_batch, offpolicy_prepare_metrics = self._prepare_offpolicy_validation_batch(
                    global_step=global_step,
                    rollout_batch=rollout_batch,
                )
            metrics["time/validation_prepare_offpolicy"] = prepare_timer.last
            metrics.update(offpolicy_prepare_metrics)
        else:
            # # logger.error(f"[UV] step={global_step} BEFORE _generate_validation_batch")
            with Timer(name="validation_generate", logger=None) as gen_timer:
                validation_batch = self._generate_validation_batch(global_step, rollout_batch=rollout_batch)
            metrics["time/validation_generate"] = gen_timer.last
            # # logger.error(f"[UV] step={global_step} AFTER _generate_validation_batch")

        # 2. offload actor_infer，为 actor_train 腾出显存（compute_log_probs 会自动 load actor_train）
        if not self.pipeline_config.async_pipeline:
            # # logger.error(f"[UV] step={global_step} BEFORE actor_infer.offload_states")
            self.actor_infer.offload_states()
            # # logger.error(f"[UV] step={global_step} AFTER actor_infer.offload_states")

        # 3. 计算优势值
        # # logger.error(f"[UV] step={global_step} BEFORE _compute_validation_advantage")
        with Timer(name="validation_compute_advantage", logger=None) as adv_timer:
            if self.uv_est_mode == "off-policy":
                avg_advantage, adv_metrics = self._compute_offpolicy_validation_advantage(validation_batch)
            else:
                avg_advantage, adv_metrics = self._compute_validation_advantage(validation_batch)
            avg_advantage=-avg_advantage
        metrics.update(adv_metrics)
        metrics["time/validation_compute_advantage"] = adv_timer.last
        metrics["validation/avg_advantage"] = avg_advantage
        # # logger.error(f"[UV] step={global_step} AFTER _compute_validation_advantage avg_advantage={avg_advantage}")

        # 3. 更新 buffer 和阈值
        self._update_advantage_threshold(avg_advantage)

        metrics["validation/buffer_size"] = len(self.uv_advantage_buffer)
        if self.uv_advantage_threshold is not None:
            metrics["validation/advantage_threshold"] = self.uv_advantage_threshold
        current_acceptance_threshold = self._get_acceptance_advantage_threshold(global_step)
        metrics["validation/acceptance_advantage_threshold_current"] = current_acceptance_threshold

        if self.uv_metrics_only:
            self.uv_rollback_history.append(0.0)
            metrics["validation/metrics_only"] = 1.0
            metrics["validation/update_accepted"] = 1.0
            metrics["validation/rollback_rate_last_100"] = 0.0
            metrics["validation/lr_adaptation_rollback_rate_window"] = 0.0
            logger.info(f"Step {global_step}: UV metrics recorded without rollback or LR adaptation.")
            return True, metrics

        # 4. 判断是否接受更新
        rollback_happened = False

        if len(self.uv_advantage_buffer) < self.uv_min_steps_for_threshold:
            # 前 min_steps_for_threshold 步总是接受（收集数据）
            accepted = True
            metrics["validation/update_accepted"] = 1.0
            logger.info(
                f"Step {global_step}: Update ACCEPTED (warming up, "
                f"step {len(self.uv_advantage_buffer)}/{self.uv_min_steps_for_threshold}, "
                f"avg_advantage={avg_advantage:.6f})"
            )
        else:
            # accepted = avg_advantage >= self.uv_advantage_threshold
            accepted = avg_advantage >= current_acceptance_threshold
            # accepted = False
            metrics["validation/update_accepted"] = 1.0 if accepted else 0.0

            if accepted:
                logger.info(
                    f"Step {global_step}: Update ACCEPTED. "
                    f"avg_advantage={avg_advantage:.6f}, "
                    f"acceptance_threshold={current_acceptance_threshold:.6f}, "
                    f"acceptance_mode={self.uv_acceptance_advantage_threshold_mode}"
                )
            else:
                logger.warning(
                    f"Step {global_step}: Update REJECTED! "
                    f"avg_advantage={avg_advantage:.6f}, "
                    f"acceptance_threshold={current_acceptance_threshold:.6f}, "
                    f"acceptance_mode={self.uv_acceptance_advantage_threshold_mode}"
                )
                rollback_happened = True

                # 5. 执行回退
                # # logger.error(f"[UV] step={global_step} ROLLBACK START (sync_from_guard -> model_update -> load_infer)")

                with Timer(name="validation_rollback", logger=None) as rollback_timer:
                    # 5.1 从 guard 恢复 actor_train 的状态（模型参数 + optimizer state）
                    # # logger.error(f"[UV] step={global_step} ROLLBACK BEFORE _sync_from_guard")
                    self._sync_from_guard(global_step)
                    # # logger.error(f"[UV] step={global_step} ROLLBACK AFTER _sync_from_guard")
                    # self._compare_rollback_debug_snapshot(global_step, tag=f"step_{global_step}_before_train")

                    # 5.2 重新同步参数到 actor_infer
                    # # logger.error(f"[UV] step={global_step} ROLLBACK BEFORE model_update")
                    self.model_update(global_step)
                    # # logger.error(f"[UV] step={global_step} ROLLBACK AFTER model_update")

                    # 5.3 重新加载 actor_infer 状态
                    # # logger.error(f"[UV] step={global_step} ROLLBACK BEFORE actor_infer.load_states")
                    self.actor_infer.load_states(blocking=True)
                    # # logger.error(f"[UV] step={global_step} ROLLBACK AFTER actor_infer.load_states DONE")

                    # 5.4 回退后 offload actor_train，否则 actor_train+actor_infer 同时在 GPU 易 OOM
                    # # logger.error(f"[UV] step={global_step} ROLLBACK BEFORE actor_train.offload_states (free GPU)")
                    self.actor_train.offload_states(blocking=True)
                    # # logger.error(f"[UV] step={global_step} ROLLBACK AFTER actor_train.offload_states DONE")

                metrics["time/validation_rollback"] = rollback_timer.last

        self.uv_rollback_history.append(1.0 if rollback_happened else 0.0)
        rollback_rate_last_100 = (
            float(sum(list(self.uv_rollback_history)[-100:])) / min(len(self.uv_rollback_history), 100)
        )
        metrics["validation/rollback_rate_last_100"] = rollback_rate_last_100
        lr_adaptation_window_values = list(self.uv_rollback_history)[-self.uv_lr_adaptation_window:]
        lr_adaptation_rollback_rate = float(sum(lr_adaptation_window_values)) / len(lr_adaptation_window_values)
        metrics["validation/lr_adaptation_rollback_rate_window"] = lr_adaptation_rollback_rate
        metrics.update(self._maybe_adapt_actor_lr(global_step, lr_adaptation_rollback_rate))

        # # 4. 激进版
        # accepted = avg_advantage >= 0
        # metrics["validation/update_accepted"] = 1.0 if accepted else 0.0

        # if accepted:
        #     logger.info(
        #         f"Step {global_step}: Update ACCEPTED. "
        #     )
        # else:
        #     logger.warning(
        #         f"Step {global_step}: Update REJECTED! "
        #         f"Rolling back to previous state..."
        #     )

        #     # 5. 执行回退
        #     # logger.error(
        #         f"Step {global_step}: Starting rollback process. "
        #     )

        #     with Timer(name="validation_rollback", logger=None) as rollback_timer:
        #         # 5.1 从 guard 恢复 actor_train 的状态（模型参数 + optimizer state）
        #         self._sync_from_guard(global_step)

        #         # 5.2 重新同步参数到 actor_infer
        #         self.model_update(global_step)

        #         # 5.3 重新加载 actor_infer 状态
        #         self.actor_infer.load_states(blocking=True)

        #     metrics["time/validation_rollback"] = rollback_timer.last

        return accepted, metrics

    @torch.no_grad()
    def save_metrics(self, batch):
        def remove_leading_zeros(A, r_mask):
            B = []
            for i in range(len(A)):
                row = A[i]
                mask = (r_mask[i] != 0).to(torch.int32)
                if not mask.any():  # 如果该行全为零
                    B.append([])  # 添加空列表
                else:
                    first_non_zero = mask.argmax().item()  # 找到第一个非零元素的索引
                    B.append(row[first_non_zero:].tolist())
            return B

        res_dict = {}
        batch_size = batch.batch["old_log_probs"].shape[0]
        res_dict["log_probs2"] = remove_leading_zeros(
            batch.batch["old_log_probs2"], batch.batch["response_mask"][:, 1:]
        )
        res_dict["logprobs"] = remove_leading_zeros(
            batch.batch["old_log_probs"], batch.batch["response_mask"][:, 1:]
        )  # 这里注意batch.batch["old_log_probs"]是right pad 格式
        res_dict["new_response_mask"] = remove_leading_zeros(
            batch.batch["response_mask"][:, 1:], batch.batch["response_mask"][:, 1:]
        )
        res_dict["prompt"] = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=False)
        res_dict["response"] = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=False)
        res_dict["old_log_probs2_entropy"] = remove_leading_zeros(batch.batch["old_log_probs2_entropy"], batch.batch["response_mask"][:, 1:])
        res_dict["ref_logprobs"] = remove_leading_zeros(
            batch.batch["ref_log_probs"], batch.batch["response_mask"][:, 1:]
        )
        res_dict["values"] = batch.batch["token_level_rewards"].numpy().tolist()
        res_dict["token_rewards"] = batch.batch["token_level_rewards"].numpy().tolist()
        res_dict["reward"] = batch.batch["response_level_rewards"].numpy().tolist()
        step = batch.meta_info["global_step"]

        assert (
            len(res_dict["logprobs"]) == batch_size
        ), f"len of logprobs is : {len(res_dict['logprobs'])}, len of batch size is :{batch_size}"
        assert (
            len(res_dict["ref_logprobs"]) == batch_size
        ), f"len of ref_logprobs is : {len(res_dict['ref_logprobs'])}, len of batch size is :{batch_size}"
        assert (
            len(res_dict["log_probs2"]) == batch_size
        ), f"len of log_probs2 is : {len(res_dict['log_probs2'])}, len of batch size is :{batch_size}"

        res = []
        for i in range(batch_size):
            temp_dict = {}
            for k, _ in res_dict.items():
                temp_dict[k] = res_dict[k][i]
            temp_dict["step"] = step
            temp_dict["response_tokens"] = self.tokenizer.convert_ids_to_tokens(
                batch.batch["responses"][i], skip_special_tokens=False
            )
            res.append(temp_dict)

        file_dir = self.pipeline_config.save_logging_board_dir
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)
        # 用时间命名domain batch 文件名
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        milliseconds = str(int(time.time() * 1000) % 1000).zfill(3)
        file_name = f"data_{current_time}_{milliseconds}.jsonl"

        try:
            with open(file_dir + file_name, "a", encoding="utf-8") as f:
                for r in res:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.info(f"Writing files catch error :{e}")

    def _compute_discounted_return(self, batch: DataProto) -> Optional[torch.Tensor]:
        """
        计算 \sum gamma^t r_t，使用原始的 response level reward (scores)。
        将原始 reward 放到每个轨迹的 eos 位置，然后计算 discounted return。

        注意：不能使用 token_level_rewards，因为它是经过 reward_postprocess normalize 过的。
        """
        if batch.batch is None:
            return None

        # 使用原始的 scores（未经 norm 的 response level reward）
        if "scores" not in batch.batch.keys():
            logger.warning("_compute_discounted_return: 'scores' not found in batch, skipping")
            return None

        scores = batch.batch["scores"].float()  # (batch_size,)

        # 获取 response mask
        mask = batch.batch.get("final_response_mask", None)
        if mask is None:
            mask = batch.batch.get("response_mask", None)
            if mask is not None:
                mask = mask[:, 1:]

        if mask is None:
            logger.warning("_compute_discounted_return: no valid mask found, skipping")
            return None

        batch_size, seq_len = mask.shape

        # 创建 token level rewards，只在 eos 位置放原始 score
        token_rewards = torch.zeros(batch_size, seq_len, device=scores.device, dtype=scores.dtype)

        # 找到每个序列最后一个非零 mask 的位置（eos 位置）
        # response_mask 的最后一个 1 的位置就是 eos 位置
        eos_indices = mask.sum(dim=1).long() - 1  # (batch_size,)
        eos_indices = torch.clamp(eos_indices, min=0)

        # 在 eos 位置放置原始 score
        batch_indices = torch.arange(batch_size, device=scores.device)
        token_rewards[batch_indices, eos_indices] = scores

        # 计算 discounted return: \sum_{t=0}^{T} gamma^t * r_t
        discounts = torch.pow(
            torch.tensor(self.pipeline_config.gamma, device=token_rewards.device, dtype=token_rewards.dtype),
            torch.arange(seq_len, device=token_rewards.device, dtype=token_rewards.dtype),
        )
        while discounts.dim() < token_rewards.dim():
            discounts = discounts.unsqueeze(0)

        returns = (token_rewards * discounts * mask.float()).sum(dim=1)
        return returns

    @staticmethod
    def _to_jsonable(value: Any) -> Any:
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().tolist()
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="ignore")
        return value

    def _resolve_only_rollout_output_path(self) -> str:
        output_path = getattr(self.pipeline_config, "only_rollout_output_path", None)
        if output_path:
            return output_path
        output_dir = (
            self.pipeline_config.rollout_dump_dir
            or self.pipeline_config.save_logging_board_dir
            or os.getcwd()
        )
        return os.path.join(output_dir, "only_rollout_acc_qwen.jsonl")

    def _dump_only_rollout_acc_jsonl(self, rollout_batch: DataProto, global_step: int):
        if rollout_batch.batch is None or "scores" not in rollout_batch.batch:
            raise RuntimeError("only_rollout mode requires rollout_batch.batch['scores'].")

        scores = rollout_batch.batch["scores"].detach().float().cpu()
        non_tensor_batch = rollout_batch.non_tensor_batch or {}
        data_cnt = int(scores.shape[0])

        sample_fields_by_id: Dict[str, Dict[str, Any]] = {}
        score_sum_by_id: Dict[str, float] = defaultdict(float)
        score_count_by_id: Dict[str, int] = defaultdict(int)

        for i in range(data_cnt):
            sample_fields: Dict[str, Any] = {}
            for key, values in non_tensor_batch.items():
                value = None
                if isinstance(values, np.ndarray):
                    if i < len(values):
                        value = values[i]
                elif isinstance(values, (list, tuple)):
                    if i < len(values):
                        value = values[i]
                else:
                    try:
                        value = values[i]
                    except Exception:
                        value = None
                if value is not None:
                    sample_fields[key] = self._to_jsonable(value)

            sample_id = str(sample_fields.get("id", i))
            if sample_id not in sample_fields_by_id:
                sample_fields_by_id[sample_id] = sample_fields

            score_sum_by_id[sample_id] += float(scores[i].item())
            score_count_by_id[sample_id] += 1

        output_rows: List[Dict[str, Any]] = []
        for sample_id, base_fields in sample_fields_by_id.items():
            row = dict(base_fields)
            row.setdefault("id", sample_id)
            row["acc_qwen"] = score_sum_by_id[sample_id] / max(score_count_by_id[sample_id], 1)
            output_rows.append(row)

        output_rows.sort(key=lambda x: str(x.get("id", "")))
        output_path = self._resolve_only_rollout_output_path()
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            for row in output_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        logger.info(
            f"only_rollout dump done: step={global_step}, "
            f"samples={len(output_rows)}, responses={data_cnt}, path={output_path}"
        )

    def _generate_validation_rollout_batch(
        self, global_step: int, num_return_sequences: Optional[int] = None
    ) -> DataProto:
        if not self.val_dataset or not hasattr(self, "val_generate_scheduler"):
            raise RuntimeError(
                "only_rollout_source=validation requires validation dataset and val_generate_scheduler."
            )

        batch = DataProto()
        reward_system_config = copy.deepcopy(self.pipeline_config.reward_system_config)
        experiment_id = reward_system_config.get("experiment_id", "roll_default_task_id")
        reward_system_config["experiment_id"] = experiment_id + "_val"
        generation_config = self.pipeline_config.validation.generating_args.to_dict()
        if num_return_sequences is not None:
            generation_config["num_return_sequences"] = int(num_return_sequences)
        batch.meta_info = {
            "is_offload_states": False,
            "generation_config": generation_config,
            "global_step": global_step,
            "reward_system_config": reward_system_config,
        }
        if self.pipeline_config.generate_opt_level == 1:
            self.actor_infer.start_server(data=DataProto(meta_info=batch.meta_info))
        if self.pipeline_config.async_pipeline:
            ray.get(
                self.val_generate_scheduler.start_sampling.remote(data=batch, batch_size=len(self.val_dataset)),
                timeout=self.pipeline_config.rpc_timeout,
            )
        try:
            generate_output: DataProto = ray.get(
                self.val_generate_scheduler.get_batch.remote(data=batch, batch_size=len(self.val_dataset)),
                timeout=self.pipeline_config.rpc_timeout,
            )
        finally:
            if self.pipeline_config.generate_opt_level == 1:
                self.actor_infer.stop_server()
        generate_output.meta_info.pop("is_offload_states", None)
        return generate_output

    def _prepare_cached_rollout(self, batch: DataProto) -> Optional[DataProto]:
        """
        为下一step重放准备prompt相关的数据（cpu存储）。
        只缓存prompt相关字段，不包含response。
        """
        if batch.batch is None:
            logger.info("prepare_cached_rollout: batch.batch is None")
            return None
        if "prompts" not in batch.batch:
            logger.info(f"prepare_cached_rollout: 'prompts' missing, keys={list(batch.batch.keys())}")
            return None

        # 只缓存prompt相关字段，用prompts重建input_ids
        prompts = batch.batch["prompts"].clone()
        pad_token_id = self.tokenizer.pad_token_id
        attention_mask = (prompts != pad_token_id).long()
        position_ids = torch.clamp(torch.cumsum(attention_mask, dim=-1) - 1, min=0)

        from roll.distributed.scheduler.protocol import TensorDict
        cached_batch = TensorDict(
            {
                "input_ids": prompts,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "prompts": prompts,
            },
            batch_size=batch.batch.batch_size,
        )
        # 保留domain字段用于group_by
        if "domain" in batch.batch:
            cached_batch["domain"] = batch.batch["domain"].clone()

        # 保留non_tensor_batch中的所有字段（不只是固定的几个）
        cached_non_tensor = {}
        if batch.non_tensor_batch is not None:
            for k, v in batch.non_tensor_batch.items():
                cached_non_tensor[k] = v

        cached = DataProto(
            batch=cached_batch,
            non_tensor_batch=cached_non_tensor if cached_non_tensor else None,
            meta_info={
                "generation_config": self.get_generation_config(),
                "reward_system_config": self.pipeline_config.reward_system_config,
            },
        )
        cached.to("cpu")
        logger.info(
            f"prepare_cached_rollout: cached batch_size={cached.batch.batch_size[0]}, "
            f"keys={list(cached.batch.keys())}"
        )
        return cached

    def _reroll_previous_step(self, global_step: int, metrics_mgr: MetricsManager):
        """
        使用当前策略在上一step的prompt上重新rollout并计算J(pi_current)与上一策略的差值。
        """
        # 仅从第二步开始重放；此前或无缓存/回报时直接跳过
        if global_step < 2:
            logger.info(
                f"reroll skip: step={global_step}, "
                f"has_cache={self.prev_rollout_cache is not None}, "
                f"has_prev_return={self.prev_policy_return is not None}"
            )
            return
        if self.prev_rollout_cache is None or self.prev_policy_return is None:
            raise RuntimeError(
                f"reroll prerequisites missing: prev_rollout_cache={self.prev_rollout_cache}, "
                f"prev_policy_return={self.prev_policy_return}"
            )
        cached_batch = getattr(self.prev_rollout_cache, "batch", None)
        cached_batch_size = cached_batch.batch_size[0] if cached_batch is not None else 0
        logger.info(
            f"reroll start: current_step={global_step}, prev_step={self.prev_rollout_step}, "
            f"cached_batch_size={cached_batch_size}"
        )
        reroll_input = self.prev_rollout_cache.clone()
        reroll_input.meta_info = {
            "generation_config": self.get_generation_config(),
            "global_step": global_step,
            "is_offload_states": False,
            "reward_system_config": self.pipeline_config.reward_system_config,
        }
        # 缓存的数据已经是纯prompt，不需要特殊处理
        # domain 保持在 non_tensor_batch 中，group_by 可以从那里读取
        if reroll_input.batch is None:
            logger.warning("reroll_input.batch is None before generate")
        # 生成新的响应
        reroll_output: DataProto = self.actor_infer.generate(reroll_input)

        # 计算新的reward
        reward_refs: List[ray.ObjectRef] = []
        # 回填 non_tensor_batch 所有字段，防止 generate 丢失这些字段
        # 注意：generate 后 batch 会扩展 num_return_sequences 倍，non_tensor_batch 也需要相应扩展
        if reroll_output.non_tensor_batch is None:
            reroll_output.non_tensor_batch = {}
        if reroll_input.non_tensor_batch is not None:
            num_return_sequences = reroll_input.meta_info.get("generation_config", {}).get("num_return_sequences", 1)
            # 回填所有缓存的 non_tensor_batch 字段（domain, ground_truth, tag, extra_info 等）
            for key, value in reroll_input.non_tensor_batch.items():
                if key not in reroll_output.non_tensor_batch:
                    # 扩展以匹配 generate 后的 batch 大小
                    reroll_output.non_tensor_batch[key] = np.repeat(value, num_return_sequences)
        if "domain" not in reroll_output.non_tensor_batch:
            raise KeyError(
                f"reroll_output missing domain key in non_tensor_batch. "
                f"reroll_input non_tensor_keys={list(reroll_input.non_tensor_batch.keys()) if reroll_input.non_tensor_batch is not None else None}"
            )
        grouped = reroll_output.group_by("domain")
        for domain, domain_batch in grouped.items():
            domain_batch.meta_info.setdefault("global_step", global_step)
            domain_batch.meta_info.setdefault("reward_system_config", self.pipeline_config.reward_system_config)
            if domain in self.rewards:
                reward_refs.extend(self.rewards[domain].compute_rewards(data=domain_batch, blocking=False))
        if reward_refs:
            rewards_proto = DataProto.materialize_concat(data_refs=reward_refs)
            reroll_output.union(rewards_proto)

        # Reroll 只计算 J(π) = Σγᵗr，不需要 KL
        # 跳过 reference 和 old_log_probs 的计算，直接 mock 为零
        # （compute_token_reward 需要这些字段，但 add_token_level_kl=false 时不影响 token_level_rewards）
        response_length = reroll_output.batch["response_mask"][:, 1:].shape[1]
        reroll_output.batch["old_log_probs"] = torch.zeros(
            reroll_output.batch.batch_size[0], response_length,
            device=reroll_output.batch.device, dtype=torch.float32
        )
        reroll_output.batch["ref_log_probs"] = torch.zeros_like(reroll_output.batch["old_log_probs"])

        # 重新处理reward和mask，获得token_level_rewards
        reroll_output.batch["prompt_id"] = torch.arange(reroll_output.batch.batch_size[0], device=reroll_output.batch.device)
        batch_grouped: Dict[str, DataProto] = reroll_output.group_by("domain")
        j_list = []
        for domain, domain_batch in batch_grouped.items():
            domain_batch, _ = get_sample_level_mask(domain_batch, self.pipeline_config)
            domain_batch, _ = reward_postprocess(domain_batch, self.pipeline_config, self.running)
            domain_batch, _ = compute_token_reward(domain_batch, self.pipeline_config, self.kl_ctrl)
            returns = self._compute_discounted_return(domain_batch)
            if returns is not None:
                j_list.append(returns)
        if j_list:
            current_return = torch.cat(j_list).mean().item()
            delta = current_return - self.prev_policy_return
            metrics_mgr.add_metric("mismatch/j_return_prev_data", current_return)
            metrics_mgr.add_metric("mismatch/j_return_delta", delta)
            logger.info(
                f"reroll metrics: current_return={current_return:.6f}, "
                f"prev_return={self.prev_policy_return:.6f}, delta={delta:.6f}"
            )
        else:
            logger.info(
                "reroll metrics: j_list empty. "
                f"batch_keys={list(reroll_output.batch.keys()) if reroll_output.batch is not None else 'None'}"
            )

    def get_generation_config(self, generating_args: Optional[GeneratingArguments] = None):
        generating_args = (
            generating_args if generating_args is not None else self.actor_infer.worker_config.generating_args
        )
        generation_config = generating_args.to_dict()
        if self.pipeline_config.async_pipeline:
            generation_config["logprobs"] = 1
        return generation_config

    @torch.no_grad()
    def run(self):
        # 计算tokens per second 系统吞吐

        # 创建一个专门管理监控指标的类
        metrics_mgr = MetricsManager()
        only_rollout_enabled = bool(getattr(self.pipeline_config, "only_rollout", False))
        only_rollout_num_return_sequences = int(
            getattr(self.pipeline_config, "only_rollout_num_return_sequences", 40)
        )
        only_rollout_source = str(getattr(self.pipeline_config, "only_rollout_source", "train"))
        if only_rollout_enabled and only_rollout_num_return_sequences <= 0:
            raise ValueError("only_rollout_num_return_sequences must be positive")
        if only_rollout_enabled and only_rollout_source not in {"train", "validation"}:
            raise ValueError("only_rollout_source must be either 'train' or 'validation'")

        tps_timer = _Timer(window_size=5)
        actor_infer_timer = _Timer(window_size=5)
        actor_infer_response_timer = _Timer(window_size=5)
        actor_train_timer = _Timer(window_size=5)

        pre_step_total_time = 0
        async_sampling_started = False
        if self.pipeline_config.async_pipeline:
            for reward_cluster in self.rewards.values():
                reward_cluster.load_states()

        for global_step in range(self.pipeline_config.max_steps):
            if global_step <= self.state.step:
                global_step += 1
                continue
            logger.info(f"pipeline step {global_step} start...")

            metrics_mgr.clear_metrics()
            with tps_timer, Timer(name="step_total", logger=None) as step_total_timer:
                # if global_step > self.state.step + 1:
                logger.info(f"pre_step_total_time: {pre_step_total_time}")
                # logger.error(f"pre_step_total_time: {pre_step_total_time}")
                metrics_mgr.add_metric("time/step_total", pre_step_total_time)
                metrics_mgr.add_metrics(self._maybe_apply_baseline_lr_decay(global_step))
                batch: DataProto = DataProto(
                    meta_info={
                        "global_step": global_step,
                        "collect_unfinished": self.pipeline_config.async_pipeline,
                        "reward_system_config": self.pipeline_config.reward_system_config
                        }
                )
                # logger.error(f"offload_states")
                # 先model update，resume时不需要保存infer cluster的状态
                if self.pipeline_config.adv_estimator == "gae":
                    self.critic.offload_states(blocking=True)
                self.actor_train.offload_states(blocking=True)
                # logger.error(f"offload_states done")
                with Timer(name="step_stop_server", logger=None) as step_stop_server_timer:
                    if async_sampling_started:
                        ray.get(
                            [
                                scheduler.pause_sampling.remote(data=batch)
                                for scheduler in self.generate_schedulers.values()
                            ],
                            timeout=self.pipeline_config.rpc_timeout,
                        )
                        self.actor_infer.stop_server()
                        async_sampling_started = False
                metrics_mgr.add_metric("time/step_stop_server", step_stop_server_timer.last)
                # logger.error(f"step_stop_server done")
                with Timer(name="step_model_update", logger=None) as step_model_update_timer:
                    self.actor_infer.offload_states(blocking=True)
                    model_update_metrics: Dict = self.model_update(global_step)
                    metrics_mgr.add_metrics(model_update_metrics)
                    generation_config = self.get_generation_config()
                    if only_rollout_enabled:
                        generation_config["num_return_sequences"] = only_rollout_num_return_sequences
                    batch.meta_info["generation_config"] = generation_config
                metrics_mgr.add_metric("time/step_model_update", step_model_update_timer.last)
                # logger.error(f"step_model_update done")
                self.actor_infer.load_states(blocking=True)
                if not self.pipeline_config.async_pipeline:
                    for reward_cluster in self.rewards.values():
                        reward_cluster.load_states()
                if only_rollout_enabled and only_rollout_source == "validation":
                    validation_batch = self._generate_validation_rollout_batch(
                        global_step=global_step,
                        num_return_sequences=only_rollout_num_return_sequences,
                    )
                    self._dump_only_rollout_acc_jsonl(validation_batch, global_step=global_step)
                    logger.info("only_rollout(validation) enabled: exit pipeline after one validation rollout.")
                    logger.info("pipeline complete!")
                    return
                # logger.error(f"load_states done")
                # 在正式收集当前step数据前，每10步用新策略重放上一step的prompt并计算J差值（delta 仍为当前步与上一步）；GAE 不做 reroll
                if self.pipeline_config.adv_estimator != "gae" and global_step % 800 == 0:
                    self._reroll_previous_step(global_step=global_step, metrics_mgr=metrics_mgr)
                # logger.error(f"reroll_previous_step done")
                if self.val_dataset and global_step % self.pipeline_config.eval_steps == 0:
                    with Timer(name="val_step", logger=None) as val_step_timer:
                        val_metrics = self.val(global_step=global_step)
                    metrics_mgr.add_metrics(val_metrics)
                    metrics_mgr.add_metric("time/val_step", val_step_timer.last)
                # logger.error(f"val_step done")
                # 要按domain group by生成对应的batch
                with (
                    actor_infer_timer,
                    actor_infer_response_timer,
                    Timer(name="step_generate", logger=None) as step_generate_timer,
                ):
                    domain_batches = {}
                    if self.pipeline_config.generate_opt_level == 1:
                        self.actor_infer.start_server(data=DataProto(meta_info=batch.meta_info))
                        batch.meta_info["is_offload_states"] = False
                    if self.pipeline_config.async_pipeline:
                        start_refs = [
                            scheduler.start_sampling.remote(
                                data=batch,
                                batch_size=self.domain_batch_size[domain],
                            )
                            for domain, scheduler in self.generate_schedulers.items()
                        ]
                        ray.get(start_refs, timeout=self.pipeline_config.rpc_timeout)
                        async_sampling_started = True
                    scheduler_refs = {}
                    for domain, scheduler in self.generate_schedulers.items():
                        scheduler_refs[domain] = scheduler.get_batch.remote(
                            data=batch, batch_size=self.domain_batch_size[domain]
                        )
                    for domain, scheduler_ref in scheduler_refs.items():
                        domain_batch: DataProto = ray.get(scheduler_ref, timeout=self.pipeline_config.rpc_timeout)
                        metrics_mgr.add_domain_metrics(
                            domain, reduce_metrics(domain_batch.meta_info.pop("metrics", {}))
                        )
                        domain_batches[domain] = domain_batch
                    generate_output = DataProto.concat([domain_batch for domain_batch in domain_batches.values()])
                    dump_rollout_to_specific_path(self.pipeline_config.rollout_dump_dir, global_step, generate_output, self.tokenizer)
                    generate_output.meta_info.pop("is_offload_states", None)
                    # logger.error(f"generate_output done")
                    # 记录当前step用于下个step重放的基础数据（GAE 不做 reroll，不缓存）
                    if self.pipeline_config.adv_estimator != "gae":
                        cached_rollout_for_next_step = self._prepare_cached_rollout(generate_output)
                    # logger.error(f"cached_rollout_for_next_step done")
                    if not self.pipeline_config.async_pipeline:
                        if self.pipeline_config.generate_opt_level == 1:
                            gen_metrics = self.actor_infer.stop_server()
                            metrics_mgr.add_metrics(reduce_metrics(gen_metrics.meta_info.pop("metrics", {})))
                        self.actor_infer.offload_states()
                        for reward_cluster in self.rewards.values():
                            reward_cluster.offload_states()
                metrics_mgr.add_metric("time/step_generate", step_generate_timer.last)
                # logger.error(f"step_generate done")
                batch = generate_output
                batch.meta_info["global_step"] = global_step
                batch.meta_info["_broadcast_non_tensor_batch"] = True
                if only_rollout_enabled and only_rollout_source == "train":
                    self._dump_only_rollout_acc_jsonl(batch, global_step=global_step)
                    logger.info("only_rollout(train) enabled: exit pipeline after first rollout.")
                    if async_sampling_started:
                        ray.get(
                            [
                                scheduler.pause_sampling.remote(data=batch)
                                for scheduler in self.generate_schedulers.values()
                            ],
                            timeout=self.pipeline_config.rpc_timeout,
                        )
                        self.actor_infer.stop_server()
                    logger.info("pipeline complete!")
                    return
                batch.non_tensor_batch['sample_uuid'] = np.array([str(uuid.uuid4()) for _ in range(batch.batch.shape[0])], dtype=object)
                # logger.error(f"batch done")

                with Timer(name="cal_ref_log_probs", logger=None) as cal_ref_log_probs_timer:
                    if self.pipeline_config.enable_reference:
                        worker_config = self.pipeline_config.reference if self.use_ref_model else self.pipeline_config.actor_train
                        worker = self.reference if self.use_ref_model else self.actor_train
                        if worker_config.use_dynamic_batching_in_infer:
                            batch, dynamic_batching_metrics = dynamic_batching_shard(
                                batch,
                                worker.dp_size,
                                worker_config.max_tokens_per_microbatch_in_infer,
                                worker_config.sequence_length_round_in_infer,
                                "reference/compute_log_probs",
                            )
                            metrics_mgr.add_metrics(dynamic_batching_metrics)
                        if not self.use_ref_model:
                            batch.meta_info["disable_adapter"] = True
                            batch.meta_info["is_offload_states"] = False
                            ref_log_probs = self.actor_train.compute_log_probs(batch, blocking=True)
                        else:
                            ref_log_probs = self.reference.compute_log_probs(batch, blocking=True)
                        metrics_mgr.add_reduced_metrics(ref_log_probs.meta_info.pop("metrics", {}))
                        ref_log_probs.rename(old_keys="log_probs", new_keys="ref_log_probs")
                        batch = batch.union(ref_log_probs)
                metrics_mgr.add_metric("time/ref_log_probs_values", cal_ref_log_probs_timer.last)
                # logger.error(f"cal_ref_log_probs done")
                with Timer(name="cal_old_log_probs_values", logger=None) as cal_old_logpb_timer:
                    if self.pipeline_config.enable_reference and not self.use_ref_model:
                        batch.meta_info["disable_adapter"] = False
                    batch.meta_info["is_offload_states"] = False
                    if self.pipeline_config.adv_estimator == "gae":
                        values_refs: List[ray.ObjectRef] = self.critic.compute_values(batch, blocking=False)

                    if self.pipeline_config.enable_old_logprobs_recompute:
                        if self.pipeline_config.actor_train.use_dynamic_batching_in_infer:
                            batch, dynamic_batching_metrics = dynamic_batching_shard(
                                batch,
                                self.actor_train.dp_size,
                                self.pipeline_config.actor_train.max_tokens_per_microbatch_in_infer,
                                self.pipeline_config.actor_train.sequence_length_round_in_infer,
                                "actor_train/compute_log_probs",
                            )
                            metrics_mgr.add_metrics(dynamic_batching_metrics)
                        old_log_probs_refs: List[ray.ObjectRef] = self.actor_train.compute_log_probs(batch, blocking=False)
                        old_log_probs = DataProto.materialize_concat(data_refs=old_log_probs_refs)
                        # logger.error(f"old_log_probs done")
                        # Customize_logging metrics, Double check call twice
                        if self.pipeline_config.save_logging_board_dir:
                            old_log_probs_refs2: List[ray.ObjectRef] = self.actor_train.compute_log_probs(
                                batch, blocking=False
                            )
                            old_log_probs2 = DataProto.materialize_concat(data_refs=old_log_probs_refs2)
                            batch.batch["old_log_probs2"] = old_log_probs2.batch["log_probs"]
                            batch.batch["old_log_probs2_entropy"] = old_log_probs2.batch["entropy"]

                        agg_entropy = agg_loss(
                            loss_mat=old_log_probs.batch["entropy"],
                            loss_mask=batch.batch["response_mask"][:, 1:],
                            loss_agg_mode="token-mean",
                        )
                        batch.meta_info["agg_entropy"] = agg_entropy
                        # logger.error(f"agg_entropy done")
                        batch.batch["old_log_probs"] = old_log_probs.batch["log_probs"]
                        metrics_mgr.add_reduced_metrics(old_log_probs.meta_info.pop("metrics", {}))
                    else:
                        # Use zeros when optimization is enabled
                        batch.batch["old_log_probs"] = torch.zeros_like(batch.batch["attention_mask"][:, 1:])
                    # logger.error(f"old_log_probs done")
                    if self.pipeline_config.adv_estimator == "gae":
                        values = DataProto.materialize_concat(data_refs=values_refs)
                        batch = batch.union(values)
                        metrics_mgr.add_reduced_metrics(values.meta_info.pop("metrics", {}))
                    # logger.error(f"values done")
                    # Mock ref_log_probs using old_log_probs if reference is disabled
                    if not self.pipeline_config.enable_reference:
                        batch.batch["ref_log_probs"] = batch.batch["old_log_probs"].clone()
                metrics_mgr.add_metric("time/old_log_probs", cal_old_logpb_timer.last)
                # logger.error(f"cal_old_log_probs done")
                # 要按domain group by处理reward
                batch.batch["prompt_id"] = torch.arange(batch.batch.batch_size[0], device=batch.batch.device)
                batch_grouped: Dict[str, DataProto] = batch.group_by("domain")
                batch_list = []
                for domain, domain_batch in batch_grouped.items():
                    # 1. 处理mask相关策略， 获取sample level mask
                    with Timer(name="get_sample_level_mask", logger=None) as get_sample_level_mask_timer:
                        domain_batch, mask_metrics = get_sample_level_mask(domain_batch, self.pipeline_config)
                        metrics_mgr.add_domain_metrics(domain, mask_metrics)
                    metrics_mgr.add_domain_metrics(domain, {"time/get_sample_level_mask": get_sample_level_mask_timer.last})
                    # logger.error(f"get_sample_level_mask done")
                    # 2. 处理reward相关策略
                    with Timer(name="reward_postprocess", logger=None) as reward_postprocess_timer:
                        domain_batch, response_level_metrics = reward_postprocess(
                            domain_batch, self.pipeline_config, self.running
                        )
                        metrics_mgr.add_domain_metrics(domain, response_level_metrics)
                    metrics_mgr.add_domain_metrics(domain, {"time/reward_postprocess": reward_postprocess_timer.last})
                    # logger.error(f"reward_postprocess done")
                    # 3. 计算token level rewards
                    with Timer(name="get_token_reward", logger=None) as get_token_reward_timer:
                        domain_batch, token_level_metrics = compute_token_reward(
                            domain_batch, self.pipeline_config, self.kl_ctrl
                        )
                        metrics_mgr.add_domain_metrics(domain, token_level_metrics)
                    metrics_mgr.add_domain_metrics(domain, {"time/get_token_reward": get_token_reward_timer.last})
                    # logger.error(f"get_token_reward done")
                    # 4. 计算advantage
                    final_response_mask = domain_batch.batch["final_response_mask"].clone()
                    with Timer(name="compute_advantage", logger=None) as compute_advantage_timer:
                        domain_batch = compute_advantage(
                            data=domain_batch,
                            gamma=self.pipeline_config.gamma,
                            lambd=self.pipeline_config.lambd,
                            adv_estimator=self.pipeline_config.adv_estimator,
                            advantage_clip=self.pipeline_config.advantage_clip,
                            whiten_advantages=self.pipeline_config.whiten_advantages,
                            whiten_rewards=self.pipeline_config.whiten_rewards,
                            response_mask=final_response_mask,
                            pipeline_config=self.pipeline_config,
                        )
                        domain_metrics = reduce_metrics(domain_batch.meta_info.pop("metrics", {}))
                        metrics_mgr.add_domain_metrics(domain, domain_metrics)
                        batch_list.append(domain_batch)
                    metrics_mgr.add_domain_metrics(domain, {"time/compute_advantage": compute_advantage_timer.last})
                    if self.pipeline_config.save_logging_board_dir:
                        self.save_metrics(domain_batch)
                    # logger.error(f"compute_advantage done")
                batch = DataProto.concat(batch_list)
                dump_batch_to_reward_system(batch, self.tokenizer)
                # logger.error(f"dump_batch_to_reward_system done")
                if batch.batch["final_response_mask"].sum() == 0:
                    logger.info("Warning: final_response_mask.sum() == 0! Current step will be skipped.")
                    metrics_mgr.add_metric("mask/final_mask_sum_eq_0", 1)
                    metrics = metrics_mgr.get_metrics()
                    # do ckpt
                    self.state.step = global_step
                    self.state.log_history.append(metrics)
                    for domain, scheduler in self.generate_schedulers.items():
                        self.state.kv[f"scheduler_state_{domain}"] = ray.get(scheduler.get_scheduler_state.remote())
                    self.do_checkpoint(global_step=global_step)
                    self.tracker.log(values=metrics, step=global_step)
                    # 即便跳过训练也保留重放数据（GAE 不做 reroll）
                    if self.pipeline_config.adv_estimator != "gae":
                        self.prev_rollout_cache = cached_rollout_for_next_step if "cached_rollout_for_next_step" in locals() else None
                        self.prev_rollout_step = global_step
                    continue
                else:
                    metrics_mgr.add_metric("mask/final_mask_sum_eq_0", 0)
                # logger.error(f"final_response_mask done")
                batch.reorder(indices=torch.argsort(batch.batch["prompt_id"]))
                batch.pop("prompt_id")
                # logger.error(f"reorder done")
                # 计算当前策略在当前step上的J(\pi)（仅非 GAE 时更新 prev 供 reroll 用）
                current_returns = self._compute_discounted_return(batch)
                if current_returns is not None:
                    current_return_mean = current_returns.mean().item()
                    metrics_mgr.add_metric("policy/j_return_current", current_return_mean)
                    if self.pipeline_config.adv_estimator != "gae":
                        self.prev_policy_return = current_return_mean
                        self.prev_rollout_step = global_step
                # logger.error(f"compute_discounted_return done")
                metrics_mgr.add_all_metrics(
                    global_step,
                    batch,
                    resource_manager=self.resource_manager,
                    actor_infer=self.actor_infer,
                    actor_train=self.actor_train,
                )
                batch_grouped: Dict[str, DataProto] = batch.group_by("domain")
                metrics_mgr.add_domain_all_metrics(global_step, batch_grouped)
                # logger.error(f"add_domain_all_metrics done")
                if self.update_validation_enabled and self.uv_est_mode == "off-policy":
                    self._push_offpolicy_rollout_history(global_step=global_step, rollout_batch=batch)
                # ===== Update Validation: 保存状态到 guard（训练更新前）=====
                # 在 Megatron 模型更新之前，保存当前模型状态到 guard
                if (
                    (self.update_validation_enabled and not self.uv_metrics_only)
                    or self.uv_log_update_norm
                    or self.random_rollback_enable
                ):
                    with Timer(name="save_guard_state", logger=None) as guard_state_timer:
                        self._sync_to_guard(global_step)
                    metrics_mgr.add_metric("time/save_guard_state", guard_state_timer.last)
                    # self._capture_rollback_debug_snapshot(global_step, tag=f"step_{global_step}_before_train")
                # # logger.error(f"save_guard_state done")
                # ===== 训练更新 Megatron 模型 =====
                with Timer(name="step_train", logger=None) as step_train_timer:
                    if self.pipeline_config.adv_estimator == "gae":
                        critic_train_metrics_refs: List[ray.ObjectRef] = self.critic.train_step(batch, blocking=False)
                    # logger.error(f"critic_train_step done")
                    with actor_train_timer:
                        # implement critic warmup
                        if self.pipeline_config.critic_warmup <= global_step:
                            # update actor
                            if self.pipeline_config.actor_train.use_dynamic_batching_in_train:
                                batch, dynamic_batching_metrics = dynamic_batching_shard(
                                    batch,
                                    self.actor_train.dp_size,
                                    self.pipeline_config.actor_train.max_tokens_per_microbatch_in_train,
                                    self.pipeline_config.actor_train.sequence_length_round_in_train,
                                    "actor_train/train_step",
                                )
                                metrics_mgr.add_metrics(dynamic_batching_metrics)
                            actor_train_metrics_refs = self.actor_train.train_step(batch, blocking=False)
                            actor_train_metrics: DataProto = DataProto.materialize_concat(
                                data_refs=actor_train_metrics_refs
                            )
                            metrics_mgr.add_reduced_metrics(actor_train_metrics.meta_info.pop("metrics", {}))
                    # # logger.error(f"actor_train_step done")
                    if self.pipeline_config.adv_estimator == "gae":
                        critic_train_metrics = DataProto.materialize_concat(data_refs=critic_train_metrics_refs)
                        metrics_mgr.add_reduced_metrics(critic_train_metrics.meta_info.pop("metrics", {}))
                    # # logger.error(f"critic_train_metrics done")
                metrics_mgr.add_metric("time/step_train", step_train_timer.last)
                if self.uv_log_update_norm:
                    with Timer(name="validation_update_norm", logger=None) as update_norm_timer:
                        update_norm_metrics = self._collect_update_norm_metrics(global_step)
                    metrics_mgr.add_metrics(update_norm_metrics)
                    metrics_mgr.add_metric("time/validation_update_norm", update_norm_timer.last)

                # ===== Random Rollback: 按固定概率随机回退本次更新（不做 update validation）=====
                if self.random_rollback_enable:
                    # 朴素抽签：randint(1, 100) <= prob*100 即触发 rollback
                    threshold = int(round(self.random_rollback_prob * 100))
                    draw = self._random_rollback_rng.randint(1, 100)
                    do_rollback = draw <= threshold
                    self.uv_rollback_history.append(1.0 if do_rollback else 0.0)
                    metrics_mgr.add_metric("validation/random_rollback_prob", self.random_rollback_prob)
                    metrics_mgr.add_metric("validation/update_accepted", 0.0 if do_rollback else 1.0)
                    metrics_mgr.add_metric(
                        "validation/rollback_rate_last_100",
                        float(sum(list(self.uv_rollback_history)[-100:])) / min(len(self.uv_rollback_history), 100),
                    )
                    if do_rollback:
                        logger.warning(
                            f"Step {global_step}: RANDOM ROLLBACK triggered (p={self.random_rollback_prob})."
                        )
                        with Timer(name="random_rollback", logger=None) as rb_timer:
                            # 仅恢复 actor_train 权重 + optimizer state；
                            # 下一步 step 开头会照常 model_update + load_infer，所以这里不需要再同步到 vLLM。
                            self._sync_from_guard(global_step)
                        metrics_mgr.add_metric("time/random_rollback", rb_timer.last)
                    else:
                        logger.info(
                            f"Step {global_step}: random rollback NOT triggered (p={self.random_rollback_prob})."
                        )

                # ===== Update Validation: 同步参数到 vLLM 并验证更新 =====
                if self.update_validation_enabled:
                    # # logger.error(f"[UV] step={global_step} ENTER update_validation block")
                    # 1. 训练完成后，同步新参数到 vLLM（需要 actor_train model 在 GPU）
                    with Timer(name="validation_model_update", logger=None) as validation_model_update_timer:
                        model_update_metrics: Dict = self.model_update(global_step)
                        metrics_mgr.add_metrics(model_update_metrics)
                    metrics_mgr.add_metric("time/validation_model_update", validation_model_update_timer.last)
                    # # logger.error(f"[UV] step={global_step} AFTER model_update (sync to vLLM)")

                    # 2. offload actor_train，为 actor_infer 腾出显存
                    # # logger.error(f"[UV] step={global_step} BEFORE actor_train.offload_states")
                    self.actor_train.offload_states(blocking=True)
                    # # logger.error(f"[UV] step={global_step} AFTER actor_train.offload_states")

                    # 3. 重新加载 actor_infer 状态以使用新参数
                    # # logger.error(f"[UV] step={global_step} BEFORE actor_infer.load_states (first time)")
                    self.actor_infer.load_states(blocking=True)
                    # # logger.error(f"[UV] step={global_step} AFTER actor_infer.load_states (first time)")

                    # 3. 验证更新并可能回退
                    # # logger.error(f"[UV] step={global_step} BEFORE _validate_and_maybe_rollback")
                    with Timer(name="validation_total", logger=None) as validation_timer:
                        update_accepted, validation_metrics = self._validate_and_maybe_rollback(global_step, rollout_batch=batch)
                        metrics_mgr.add_metrics(validation_metrics)
                    metrics_mgr.add_metric("time/validation_total", validation_timer.last)
                    # # logger.error(f"[UV] step={global_step} AFTER _validate_and_maybe_rollback accepted={update_accepted}")

                    # 4. 如果验证不通过，回退流程已经在 _validate_and_maybe_rollback 中完成：
                    #    - 从 guard 恢复 actor_train（Megatron）
                    #    - 同步参数到 actor_infer（vLLM）
                    #    - 重新加载 actor_infer 状态
                    #    如果回退，跳过后续步骤，直接进入下一个 step
                    if not update_accepted:
                        # 记录 metrics 并 continue 到下一个 step
                        metrics = metrics_mgr.get_metrics()
                        metrics["system/step"] = global_step
                        self.state.step = global_step
                        self.state.log_history.append(metrics)

                        for domain, scheduler in self.generate_schedulers.items():
                            self.state.kv[f"scheduler_state_{domain}"] = ray.get(scheduler.get_scheduler_state.remote())

                        self.tracker.log(values=metrics, step=global_step)
                        # logger.error(f"Step {global_step}: Update rejected, skipping to next step")
                        global_step += 1
                        continue

                tps_timer.push_units_processed(n=torch.sum(batch.batch["attention_mask"]).detach().item())
                actor_infer_timer.push_units_processed(n=torch.sum(batch.batch["attention_mask"]).detach().item())
                actor_infer_response_timer.push_units_processed(
                    n=torch.sum(batch.batch["response_mask"]).detach().item()
                )
                actor_train_timer.push_units_processed(n=torch.sum(batch.batch["attention_mask"]).detach().item())

                for domain, scheduler in self.generate_schedulers.items():
                    self.state.kv[f"scheduler_state_{domain}"] = ray.get(scheduler.get_scheduler_state.remote())

                metrics = metrics_mgr.get_metrics()
                # do ckpt
                self.state.step = global_step
                self.state.log_history.append(metrics)

                # 更新用于下一step重放的数据（GAE 不做 reroll）
                if self.pipeline_config.adv_estimator != "gae":
                    self.prev_rollout_cache = cached_rollout_for_next_step if "cached_rollout_for_next_step" in locals() else None

                self.do_checkpoint(global_step=global_step)

                self.tracker.log(values=metrics, step=global_step)

                if global_step % self.pipeline_config.logging_steps == 0:
                    if int(os.environ.get("RAY_PROFILING", "0")):
                        timeline_dir = os.path.join(self.pipeline_config.profiler_output_dir, "timeline")
                        os.makedirs(timeline_dir, exist_ok=True)
                        ray.timeline(
                            filename=os.path.join(timeline_dir, f"timeline-step-{global_step}.json"),
                        )

                    prompts = self.tokenizer.batch_decode(generate_output.batch["prompts"], skip_special_tokens=True)
                    responses = self.tokenizer.batch_decode(
                        generate_output.batch["responses"], skip_special_tokens=True
                    )
                    generate_examples = [{"prompt": p, "response": r} for p, r in zip(prompts, responses)][:10]
                    logger.info(json.dumps(generate_examples, ensure_ascii=False))
                    logger.info(json.dumps(metrics, ensure_ascii=False))

                logger.info(f"pipeline step {global_step} finished")
                global_step += 1
            pre_step_total_time = step_total_timer.last

        if async_sampling_started:
            final_batch = DataProto(meta_info={"global_step": self.pipeline_config.max_steps})
            ray.get(
                [
                    scheduler.pause_sampling.remote(data=final_batch)
                    for scheduler in self.generate_schedulers.values()
                ],
                timeout=self.pipeline_config.rpc_timeout,
            )
            self.actor_infer.stop_server()

        logger.info("pipeline complete!")

    @torch.no_grad()
    def val(self, global_step):
        val_metrics_mgr = MetricsManager()
        batch = DataProto()

        with Timer(name="step_generate", logger=None) as step_generate_timer:
            reward_system_config = copy.deepcopy(self.pipeline_config.reward_system_config)
            experiment_id = reward_system_config.get("experiment_id", 'roll_default_task_id')
            # TODO: 当前通过在 experiment_id 后加上_val 的方式来区分 reward system train 和 val 的日志记录, 后续考虑更优雅的 train 和 val 区分方式。
            reward_system_config["experiment_id"] = experiment_id + '_val'
            batch.meta_info = {
                "is_offload_states": False,
                "generation_config": self.pipeline_config.validation.generating_args.to_dict(),
                "global_step": global_step,
                "reward_system_config": reward_system_config
            }

            if self.pipeline_config.generate_opt_level == 1:
                self.actor_infer.start_server(data=DataProto(meta_info=batch.meta_info))
            if self.pipeline_config.async_pipeline:
                ray.get(
                    self.val_generate_scheduler.start_sampling.remote(
                        data=batch,
                        batch_size=len(self.val_dataset),
                    ),
                    timeout=self.pipeline_config.rpc_timeout,
                )
            try:
                generate_output: DataProto = ray.get(
                    self.val_generate_scheduler.get_batch.remote(data=batch, batch_size=len(self.val_dataset)),
                    timeout=self.pipeline_config.rpc_timeout,
                )
            finally:
                if self.pipeline_config.generate_opt_level == 1:
                    self.actor_infer.stop_server()

            generate_output.meta_info.pop("is_offload_states", None)
            val_metrics_mgr.add_metric("time/step_generate", step_generate_timer.last)

        batch = generate_output
        val_correct_mean = (batch.batch["scores"] == 1).detach().float().mean().item()
        val_metrics_mgr.add_metric("val_correct/all/mean", val_correct_mean)
        logger.info(json.dumps({"val_correct/all/mean": val_correct_mean}, ensure_ascii=False))

        epoch_batch = batch.pop(batch_keys=["scores"], non_tensor_batch_keys=["tag"])

        grouped_batch = epoch_batch.group_by("tag")
        for group_key, group_batch in grouped_batch.items():
            score_mean = group_batch.batch["scores"].mean().item()
            logger.info(f"val_correct/{group_key}:  {score_mean}")
            val_metrics_mgr.add_domain_metrics(
                "val_correct", {f"{group_key}/mean": (group_batch.batch["scores"] == 1).detach().float().mean().item()}
            )

        return val_metrics_mgr.get_metrics()
