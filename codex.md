# Codex Debug Handoff

Last updated: 2026-07-26 (Asia/Hong_Kong)

## Purpose

Read this file before continuing work on this repository. Keep it updated after
each debugging session so a restarted agent can resume from verified facts.

## Current Goal

Run the Qwen3 RLVR Megatron example from the repository root:

```bash
bash examples/qwen3-8B-rlvr_megatron/run_rlvr_pipeline.sh
```

Expected Conda environment: `roll-qwen3-rlvr`.

## Debug History

### 1. Python could not import `roll`

Observed error:

```text
ModuleNotFoundError: No module named 'roll'
```

Confirmed cause:

- `roll/` exists in the repository.
- Running `python examples/start_rlvr_pipeline.py` puts `examples/`, rather than
  the repository root, at the front of Python's import path.
- The active environment did not expose the repository package to that script.

Immediate workaround:

```bash
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
```

Durable alternatives:

```bash
python -m pip install -e . --no-deps
```

or add this after `cd "${ROOT_DIR}"` in the launcher:

```bash
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
```

Status: the import progressed past `roll`, so the `PYTHONPATH` workaround was
effective. No permanent launcher change was made by Codex.

### 2. Incompatible Transformers version

Observed error:

```text
ImportError: cannot import name 'AutoModelForVision2Seq' from 'transformers'
```

Confirmed facts:

- The Conda environment had `transformers==5.14.1` installed.
- `hasattr(transformers, "AutoModelForVision2Seq")` returned `False`.
- `roll/models/model_providers.py` imports `AutoModelForVision2Seq`.
- The repository's Torch 2.6 SGLang requirements pin
  `transformers==4.51.1`; the Torch 2.6 vLLM requirements do not currently pin
  a Transformers version.

Recommended fix:

```bash
python -m pip install "transformers==4.51.1"
```

Verification:

```bash
python -c 'import transformers; from transformers import AutoModelForVision2Seq; print(transformers.__version__)'
```

Expected output: `4.51.1` with no import error.

Status: confirmed fixed. The environment now has `transformers==4.51.1`, and
startup progressed past the original `AutoModelForVision2Seq` import error.

### 3. FlashAttention binary ABI mismatch

Observed error while importing `transformers.PreTrainedModel`:

```text
flash_attn_2_cuda.cpython-310-x86_64-linux-gnu.so: undefined symbol:
_ZN3c105ErrorC2ENS_14SourceLocationENSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEv
```

Confirmed environment versions:

- `torch==2.6.0`
- `torchvision==0.21.0`
- `torchaudio==2.6.0`
- `vllm==0.8.4`
- `transformers==4.51.1`
- `flash_attn==2.8.3.post1`
- `transformer_engine==2.2.0`

Diagnosis: this is specifically a C++11 ABI mismatch, despite the custom
wheel filename claiming `cxx11abiFALSE`:

- Runtime PyTorch is `2.6.0+cu124` and
  `torch._C._GLIBCXX_USE_CXX11_ABI` is `False`.
- The installed extension has the same SHA-256 as the `.so` inside the user's
  custom wheel (`685413fa...745a49`), so this is not an older leftover `.so`.
- The extension requires `c10::Error(..., std::__cxx11::basic_string...)`, which
  is the new/TRUE ABI.
- PyTorch's `libc10.so` exports `c10::Error(..., std::basic_string...)`, which is
  the old/FALSE ABI.

Therefore the wheel is internally ABI TRUE even though its filename says ABI
FALSE. Likely build-time causes include a reused native build directory, a build
isolation environment different from the target environment, or a forced
compiler ABI flag. No FlashAttention source tree, Ninja cache, or relevant ABI
environment variable was present under `/mnt/shared-storage-user/liujinyi/lj`
when inspected, so the original build command/log is still needed to distinguish
those causes. The current `requirements_torch260_vllm.txt` no longer includes
`flash-attn`, and the Qwen3 RLVR config does not explicitly select the Hugging
Face `fa2` attention backend.

Recommended first fix:

```bash
python -m pip uninstall -y flash-attn
```

Verification:

```bash
python -c 'from transformers import PreTrainedModel; print("transformers import OK")'
```

Only if a later error proves FlashAttention is required, rebuild/reinstall a
version against the already-installed `torch==2.6.0`; do not reinstall vLLM or
upgrade PyTorch as the first response.

Status: fixed in the current container. `flash-attn` is now
`2.7.4.post1` instead of the incompatible custom `2.8.3.post1` wheel, and the
final import self-check succeeds. The exact install command is not recorded.

### 4. Hydra config path contains `examples/` twice

Observed error:

```text
hydra.errors.MissingConfigException: Primary config directory not found.
.../ROLL/examples/examples/qwen3-8B-rlvr_megatron
```

Confirmed cause: both local launchers pass
`--config_path examples/qwen3-8B-rlvr_megatron`, but Hydra resolves the config
path relative to `examples/start_rlvr_pipeline.py`. This produces the invalid
`examples/examples/...` path.

Required correction in both launchers:

```text
--config_path qwen3-8B-rlvr_megatron
```

Affected files:

- `examples/qwen3-8B-rlvr_megatron/run_rlvr_pipeline.sh`
- `examples/qwen3-8B-rlvr_megatron/run_rlvr_pipeline_2gpus.sh`

Status: fixed in the four-GPU launcher, which now passes
`--config_path qwen3-8B-rlvr_megatron`. The untracked two-GPU launcher still has
the incorrect `examples/qwen3-8B-rlvr_megatron` value.

### 5. Ray CLI is incompatible with the installed Click version

Observed behavior: ROLL ran `ray start --head ...`, but the `ray` executable
failed during import with:

```text
ValueError: <object object at ...> is not a valid Sentinel
```

Confirmed facts:

- `ray==2.47.1`
- `click==8.4.2`
- `typer==0.27.0`
- Even `ray --version` reproduces the same traceback, before Ray parses ports,
  addresses, or cluster options.
- Ray's package metadata only declares `click>=7.0`, but this Ray CLI deep-copies
  Click command objects and is not compatible with the newer Click `Sentinel`.
- Installed `vllm==0.8.4` accepts `ray[cgraph]>=2.43.0` except `2.44.*`, so Ray
  does not need to be reinstalled for this error.

Recommended minimal fix:

```bash
python -m pip install "click==8.1.8"
ray --version
```

Status: fixed in the current container. `click==8.1.8` is installed and
`ray --version` succeeds with Ray 2.47.1.

### 6. Transformer Engine cannot find pip-installed NVIDIA libraries

Observed in `ActorWorker(reference-0)`:

```text
OSError: libcudnn_adv.so.9: cannot open shared object file
```

Confirmed facts:

- `nvidia-cudnn-cu12==9.1.0.70` is installed.
- `libcudnn_adv.so.9` exists under
  `$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cudnn/lib`.
- Direct Transformer Engine import also initially failed on `libcublas.so.12`.
- The shell's `LD_LIBRARY_PATH` only contained `/usr/local/nvidia/lib` and
  `/usr/local/nvidia/lib64`, not the pip NVIDIA library directories.
- Prepending all `$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/*/lib`
  directories made `transformer_engine==2.2.0` import successfully.

Diagnosis: cuDNN is installed, but Transformer Engine's dynamic loader cannot
see the pip-installed NVIDIA shared-library directories. This is a library
search-path issue, not a missing cuDNN package.

Recommended launch preparation:

```bash
ROLL_NVIDIA_ROOT="$CONDA_PREFIX/lib/python3.10/site-packages/nvidia"
ROLL_NVIDIA_LIBS=$(find "$ROLL_NVIDIA_ROOT" -mindepth 2 -maxdepth 2 -type d -name lib -print | sort | paste -sd:)
export LD_LIBRARY_PATH="$ROLL_NVIDIA_LIBS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
python -c 'import transformer_engine; print("TE import OK", transformer_engine.__version__)'
ray stop --force
```

Restart the launcher after exporting the path so the newly started Ray cluster
and actors inherit it.

Additional runtime finding after the library-path fix:

- The pipeline completed one vLLM rollout, then reference log-prob computation
  entered Transformer Engine fused attention for the first time.
- Fused attention failed with `Unable to dlopen libcudart.so`.
- The pip CUDA runtime directory contains `libcudart.so.12` but no unversioned
  `libcudart.so` symlink.
- System CUDA 12.8 has that symlink, but mixing its runtime with the PyTorch
  `2.6.0+cu124` environment is less desirable than linking the environment's own
  CUDA runtime.

Recommended fix after stopping Ray:

```bash
ROLL_CUDART_DIR="$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cuda_runtime/lib"
ln -s libcudart.so.12 "$ROLL_CUDART_DIR/libcudart.so"
```

Keep that directory in `LD_LIBRARY_PATH`, verify with
`python -c 'import ctypes; ctypes.CDLL("libcudart.so")'`, then restart Ray and the
launcher. Use plain `ln -s` so the command fails safely instead of overwriting an
existing file.

Status: fixed in the current container. The unversioned `libcudart.so`
symlink now points to `libcudart.so.12`, and an explicit-path self-check loaded
Transformer Engine, Transformers, and `libcudart.so` successfully. Both Qwen3
launchers now add the pip NVIDIA library directories to `LD_LIBRARY_PATH` when
`CONDA_PREFIX` is set.

## Container Restart State

Verified in the current container on 2026-07-26:

- `torch==2.6.0+cu124`, C++11 ABI false
- `transformers==4.51.1`
- `vllm==0.8.4`
- `ray==2.47.1`, `click==8.1.8`
- `transformer-engine==2.2.0`
- `flash-attn==2.7.4.post1`
- `wandb==0.28.0`
- `libcudart.so -> libcudart.so.12` exists in the Conda environment
- Explicit `PYTHONPATH` and pip NVIDIA `LD_LIBRARY_PATH` self-check passes

Persistence boundaries:

- Repository changes under `/mnt/shared-storage-user/liujinyi/lj/ROLL` survive a
  new container when shared storage is remounted.
- `/root/miniconda3` is a symlink to `/data/miniconda3`, and `/data` is an XFS
  mount from `/dev/rbd1`. Package versions and the `libcudart.so` symlink should
  survive when a new container mounts the same `/data` volume.
- The Qwen3 four- and two-GPU launchers now construct `PYTHONPATH` and the pip
  NVIDIA `LD_LIBRARY_PATH` automatically. Direct invocations that bypass those
  launchers still need equivalent environment setup.
- A container attached to a different or newly initialized `/data` volume would
  still need the package pins and compatibility symlink restored.

Therefore the current persistent Conda environment is repaired for containers
that reuse this `/data` volume, but each new shell still needs a small launch
preparation block.

## Requested Configuration Changes

### W&B offline tracking

On 2026-07-26, the Qwen3 8B RLVR config was changed from TensorBoard to
Weights & Biases offline tracking:

- `track_with: wandb`
- `tracker_kwargs.mode: offline`
- `system_envs.WANDB_MODE: offline`, propagated to Ray workers
- W&B root directory: `/mnt/shared-storage-user/liujinyi/lj/ROLL`
- Effective W&B run directory: `/mnt/shared-storage-user/liujinyi/lj/ROLL/wandb`
- `/wandb/` added to the repository `.gitignore`

The configured root intentionally omits the final `wandb` component because
W&B appends its own `wandb/` directory. Hydra composition and W&B's effective
path calculation were verified without starting a training run.

Affected files:

- `examples/qwen3-8B-rlvr_megatron/rlvr_config.yaml`
- `.gitignore`

## Launcher Notes

The launcher has user-owned, uncommitted changes. Preserve them.

- It now resolves `ROOT_DIR`, changes to the repository root, defaults to GPUs
  `0,1,2,3`, and forwards command-line arguments.
- The four-GPU launcher correctly passes
  `--config_path qwen3-8B-rlvr_megatron`.
- The untracked two-GPU launcher now correctly passes
  `--config_path qwen3-8B-rlvr_megatron`.
- Both launchers export the repository `PYTHONPATH` and discover pip NVIDIA
  library directories under the active `CONDA_PREFIX`.

## Next Steps

1. For a container using a different `/data` volume, restore the pinned Python
   packages and recreate the `libcudart.so` compatibility symlink before Ray.
2. Capture any later training-time traceback and append it here.

### 7. Codex metadata directories and relaunch

Observed shell output before setup:

```text
ls: cannot access '.agents': No such file or directory
ls: cannot access '.codex': No such file or directory
```

Confirmed cause: the metadata directories did not exist at the time of the
first `ls`; `mkdir -p .agents .codex` created them. In the current managed
shell they are present as root-owned mode `0555` directories, so they are not
writable by the repository user.

Verification:

```text
codex-cli 0.145.0
WARNING: proceeding, even though we could not create PATH aliases: Read-only file system (os error 30)
```

The Codex executable is available and the relaunch command is syntactically
valid. The PATH-alias warning is caused by the managed shell's read-only
filesystem, not by a missing `.agents` or `.codex` directory. No repository
source change is required. Next step: run `codex --sandbox workspace-write
--ask-for-approval never` in the user's interactive shell and continue with the
pending launcher/runtime steps above.

### 8. Comparison of `/mnt/shared-storage-user/liujinyi/lj/roll_edit`

On 2026-07-26, the supplied edit directory was compared with the current
repository before any migration.

Confirmed file-layout mismatch:

- The user-listed set contains 24 path-qualified files, but `roll_edit` contains
  only 13 flat files and preserves no directory names.
- Twelve flat Python files map unambiguously by basename to current modules.
- `rlvr_config_4b_wrollback_agentic.yaml` has no current path counterpart.
- The listed `roll/pipeline/rlvr/math_rule_reward_worker.py` is not the current
  module path; the live module is
  `roll/pipeline/rlvr/rewards/math_rule_reward_worker.py`.
- The listed scripts, `requirements/requirements_torch280_vllm.txt`,
  `training_args.py`, both VLM pipelines, and root `__init__.py` are absent from
  `roll_edit`, so their changes cannot be verified or migrated from this copy.

Confirmed compatibility mismatch:

- Current checked-out `HEAD` is `97447137` (2026-01-28); the current repaired
  environment is `torch==2.6.0`, `vllm==0.8.4`.
- The edit set expects APIs from a newer/different code line, including
  `is_actor_infer_overlapping_with_any_cluster`, `TaskGroup`,
  `DurationTracker`, and `create_async_llm`; those definitions are absent from
  the current tree (the first three are present only in `origin/main`, while
  `create_async_llm` is also absent from the current `roll.third_party.vllm`).
- `actor_worker.py` imports the missing
  `roll.utils.dynamic_batching.get_dynamic_batching_loss_scale`.
- `rlvr_pipeline.py` imports the missing
  `dump_batch_to_reward_system` from `roll.pipeline.rlvr.utils` and the
  `is_report_data_finished` symbol from `generate_scheduler.py`; the current
  `is_report_data_finished` lives in `async_generate_scheduler.py` instead.
- `requirements_torch280_vllm.txt` in the current repo pins Torch 2.8/vLLM
  0.10.2, unlike the active Torch 2.6/vLLM 0.8.4 environment.

Confirmed edit-level risks:

- `actor_worker.py` performs arithmetic with `infer_log_probs` immediately
  after `data.batch.get("infer_logprobs")`; if the key is absent, the intended
  fallback to `old_log_probs` is commented out and training raises a type
  error.
- The same file defines configurable sequence-mask thresholds in
  `rlvr_config.py` but uses hard-coded `0.999/1.001` and `0.1/10.0` values,
  ignoring those config fields.
- `functionals.py:reward_postprocess` clamps `log_ratio_seq` even when
  `response_mask` is absent, although that branch never initializes
  `log_ratio_seq`.
- `utils.py:dump_batch_to_reward_system` leaves `values` as a Torch tensor in
  the JSON payload, which is not JSON serializable when reward-system dumping
  is enabled; it also contains a hard-coded fallback authorization token.
- The supplied 4B YAML uses `track_with: ml_tracker`, but the current tracker
  registry supports only `tensorboard`, `wandb`, `stdout`, and `swanlab`.

Verification performed:

- All 12 mapped Python files plus the YAML file were statically inspected.
- `PYTHONPYCACHEPREFIX=<temporary-dir> python -m py_compile` succeeded for all
  13 Python files.
- PyYAML parsed `rlvr_config_4b_wrollback_agentic.yaml` successfully.
- `git diff --check` found extensive trailing whitespace in the edit files;
  this is cosmetic but should be cleaned before applying a patch.

Conclusion: do not copy the edit directory directly into the current repo. The
next pending step is to obtain a complete, path-preserving edit tree (and its
matching dependency/worker-config changes), or explicitly choose the newer
`origin/main` code line as the migration base before preparing a staged patch.

## Repository Safety

The worktree was already dirty before this file was added. In particular, the
RLVR config, launcher, vLLM requirements, datasets, and a two-GPU launcher have
user-owned changes. Do not revert or overwrite them while debugging.

### 9. Repository write-access verification

On 2026-07-26, the restarted shell was checked from the repository root after
the ownership/permission commands were run.

Observed shell messages:

```text
bash: 然后重新进入: command not found
bash: 如果这两条成功了，再启动: command not found
bash: 另外，重新启动后的: command not found
```

Cause: explanatory Chinese text was pasted into Bash as commands; it is not a
repository or permission failure.

Verification from `/mnt/shared-storage-user/liujinyi/lj/ROLL`:

```bash
printf test > .write-test
rm .write-test
```

Result: `write-test: success`. The current shell can create and remove files in
the repository, so Codex has the required workspace write access. Existing
uncommitted user changes remain untouched.

The shell also had `CONDA_PREFIX=/root/miniconda3/envs/roll-qwen3-rlvr`, but
before the requested PATH export `python` resolved to `/usr/bin/python`. After
`export PATH="$CONDA_PREFIX/bin:$PATH"`, it resolved to the Conda Python
(`Python 3.10.20`).

### 10. First compatible `roll_edit` migration slice

On 2026-07-26, the edit tree was compared with detached `HEAD=97447137` and
was not copied wholesale because it references newer APIs absent from this
checkout (`create_async_llm`, update-validation helpers, and additional
pipeline symbols).

Applied compatible changes:

- `roll/datasets/chat_template.py`: disable thinking for the native template.
- `roll/pipeline/rlvr/rlvr_config.py`: add opt-in `format_reward_type` and
  `response_level_reward_type` fields, with defaults preserving the old binary
  reward behavior.
- `roll/pipeline/rlvr/rewards/math_rule_reward_worker.py`: add balanced
  `\\boxed{}` extraction, boxed format scoring, optional soft response scoring,
  boxed-aware answer parsing, and expose existing auxiliary reward tensors.
- `roll/utils/functionals.py`: merge distributed Welford metrics and add a
  partial-key-safe `reduce_metrics_list` helper.
- `tests/utils/test_functionals.py`: add coverage for both metric helpers.

Verification:

```text
Python 3.10.20 (roll-qwen3-rlvr)
5 passed, 2 warnings in tests/utils/test_functionals.py
py_compile: success
boxed helper checks: success (isolated function execution)
```

Attempting to import the reward worker directly for a runtime boxed-parser
check failed before module loading with:

```text
PermissionError: [Errno 1] Operation not permitted
```

Cause: the import-time `roll.utils.network_utils.get_node_ip()` opens a UDP
socket to `8.8.8.8`; the managed sandbox blocks that network operation. This is
an environment limitation, not a parser traceback.

Next pending step: review and port the edit tree's actor/pipeline update-
validation and async-vLLM changes only after mapping their missing dependencies
to this checkout; do not replace the current files wholesale.

### 11. Actor mismatch/TIS and update-validation migration (current session)

Observed compatibility errors during this session:

```text
/usr/bin/python: No module named pytest
PermissionError: [Errno 1] Operation not permitted
__main__.py: error: unrecognized arguments: tests/pipeline/test_rlvr_pipeline.py -q
```

The first command used the system Python instead of the active Conda
environment; exporting `$CONDA_PREFIX/bin` selected Python 3.10.20 with
`pytest==9.1.1`. The second occurred while importing the full worker module in
the restricted shell because `checkpoint_manager` calls `socket.socket()` at
import time; it is an environment restriction, not a repository traceback from
the training process. The pipeline test also cannot be collected with normal
pytest arguments because its module parses `sys.argv` at import time; no test
body ran.

Relevant verified versions:

- `torch==2.6.0+cu124`
- `vllm==0.8.4`
- Python `3.10.20`

Changes made:

- Added `ActorWorker.generate_request`, a synchronous adapter around the
  existing callback server, retaining per-token `output_logprobs` for
  validation; validation now explicitly starts and stops infer servers around
  its direct requests.
- Added the optional `pipeline_config` argument to `compute_advantage` for
  edit-side call compatibility.
- Extended the Megatron guard to snapshot/restore the underlying torch
  optimizer shard when `use_distributed_optimizer=True`; the installed
  wrapper's public `state_dict()` omits fp32 master/Adam moment tensors.
- Preserved the Qwen template contract: `qwen3` sets `enable_thinking=True`,
  while `native`/`qwen2_5` set `enable_thinking=False`.
- Migrated sequence reward IS, GRPO-IS, and SNIS statistics into
  `reward_norm`/`reward_postprocess`, with config flags and mutual-exclusion
  validation; existing actor mismatch token/sequence metrics, truncated IS,
  reverse IS, mismatch KL, ratio dumps, and dynamic-batch loss scaling remain
  in `roll/pipeline/rlvr/actor_worker.py`.
- Corrected the `get_encode_function` argument order in
  `rlvr_rollout_pipeline.py` after the pipeline migration changed its helper
  signature.

Verification:

- `python -m py_compile` succeeds for the modified functionals, config,
  worker, actor, Megatron strategy, RLVR pipeline, and rollout pipeline files.
- `tests/utils/test_functionals.py` and `tests/utils/test_dynamic_batching.py`:
  `7 passed, 2 warnings`.
- A direct reward-normalization smoke test produced finite SNIS and GRPO-IS
  outputs.

Not yet verified:

- `generate_request` has not run against a live Ray/vLLM worker in this shell.
- Megatron distributed-optimizer guard snapshots/rollback and scheduler/RNG
  restoration have not been observed in a real training worker; the current
  implementation uses the installed Megatron optimizer `state_dict()` path.
- The full Qwen3 multi-GPU pipeline has not been run; validation must still be
  dry-run tested with a tiny batch before enabling it for a long job.

Next pending step: run a minimal Ray/vLLM validation request and an isolated
Megatron guard round-trip, then fix any strategy-specific state or logprob
shape issues observed there.

### 12. Ray actor double-wrapping and scheduler API mismatch

Observed startup error:

```text
TypeError: The @ray.remote decorator must be applied to either a function or a class.
```

It occurred at both training and validation scheduler construction sites in
`RLVRPipeline.__init__` when evaluating:

```python
ray.remote(DynamicSamplingScheduler)
```

Relevant verified versions and configuration:

- Python `3.10.20`
- Ray `2.47.1`
- Torch `2.6.0+cu124`
- vLLM `0.8.4`
- The Qwen3 config composes with `async_pipeline=False`,
  `generate_opt_level=1`, and update validation enabled.

Confirmed cause:

- The current `roll.distributed.scheduler.generate_scheduler` already declares
  `DynamicSamplingScheduler` with `@ray.remote`, so the imported symbol is a
  Ray `ActorClass`, not a plain Python class. Applying `ray.remote()` a second
  time raises the reported `TypeError`.
- The migrated `rlvr_pipeline.py` used APIs from the newer, plain-class
  scheduler in `roll_edit`, while this checkout still has the older decorated
  sync/async schedulers. The mismatch also removed the required response
  callback, passed an unsupported `global_step` keyword to `get_batch`, omitted
  the vLLM request-server lifecycle, and called nonexistent scheduler
  `shutdown` methods. Fixing only the first traceback would therefore have
  exposed additional deterministic failures.

Changes made:

- Construct the already-remote scheduler with `scheduler_cls.options(...).remote(...)`.
- Restore selection of `AsyncDynamicSamplingScheduler` only when
  `pipeline_config.async_pipeline` is enabled.
- Restore response/query callbacks required by the current scheduler and pass
  `global_step` through `DataProto.meta_info`, matching its `get_batch(data,
  batch_size)` signature.
- Restore vLLM `start_server`/`stop_server` around sync and validation request
  windows, and the current async scheduler's `start_sampling`/`pause_sampling`
  lifecycle. Remove calls to scheduler methods that do not exist in this
  checkout.
- Fix the no-separate-reference branch to use `self.actor_train` (the cluster)
  instead of `self.pipeline_config.actor_train` (the config) when dynamic
  batching reads `worker.dp_size`.
- Make both Qwen3 launchers export the repository `PYTHONPATH` and pip NVIDIA
  library paths; correct the two-GPU Hydra config path.
- Change the Python entrypoint to preserve unknown CLI arguments as Hydra
  overrides. Previously `parse_args()` rejected every forwarded override, so
  the two-GPU launcher could not reach configuration composition.

Verification:

```text
DynamicSamplingScheduler and AsyncDynamicSamplingScheduler are Ray ActorClass instances
both .options(...).remote actor construction contracts are available
pipeline import and Ray actor-class contract: OK
RLVR config compose: OK (sync scheduler, generate_opt_level=1)
py_compile and compileall: success
14 passed, 6 warnings (scheduler padding + functionals + dynamic batching tests)
bash -n: both Qwen3 launchers passed
entrypoint accepted and composed the complete two-GPU override set
Transformer Engine 2.2.0, Transformers, and libcudart.so import self-check: passed
git diff --check on touched source/launchers: passed
```

The scheduler test/import process needed a process-local replacement for
`get_node_ip()` because this managed sandbox prohibits socket creation. No
network utility source was changed. A real multi-GPU pipeline run was not
possible here because the managed environment exposes no `nvidia-smi` and
blocks Ray socket setup, so the fix is not marked as end-to-end observed.

Next pending step: from the GPU shell with the `roll-qwen3-rlvr` environment
activated, run:

```bash
bash examples/qwen3-8B-rlvr_megatron/run_rlvr_pipeline.sh
```

Confirm startup passes scheduler construction and reaches the first rollout;
record any later runtime traceback here.

### 13. FP16 and online INT8 inference modes

Requested feature: support `quantization: fp16` and `quantization: int8` in
the vLLM actor-inference strategy, following the existing online FP8 behavior.

Relevant verified versions:

- Python `3.10.20`
- Torch `2.6.0+cu124`
- vLLM `0.8.4`

Confirmed implementation constraints:

- vLLM treats FP16 as a model `dtype`, not as a quantization backend. Passing
  the literal `quantization="fp16"` to vLLM would be invalid.
- ROLL's FP8 support patches weight loading so floating-point Megatron weights
  are quantized again after every online model update. An INT8 implementation
  must preserve that behavior; mapping `int8` to a serialized-checkpoint
  backend such as BitsAndBytes would not support the current dummy-load and
  online-update path.

Changes made:

- Added `roll.third_party.vllm.quantization` and registered an internal
  `roll_int8` vLLM quantization method.
- User-facing `quantization: fp16` and `float16` aliases now remove the vLLM
  quantization argument and force `dtype=float16`.
- User-facing `quantization: int8` maps to `roll_int8`, retaining FP16/BF16
  activations. Dense linear layers use symmetric per-output-channel INT8
  weights with dynamically quantized INT8 activations (W8A8 CUTLASS). Fused
  MoE layers reuse vLLM's online W8A16 INT8 expert method.
- The INT8 weight loader accepts floating-point model-update tensors,
  recomputes scales, writes CUTLASS's transposed INT8 layout, and updates the
  scale parameter on every load. Megatron-to-vLLM transfer remains in the
  source floating-point dtype; quantization happens on the infer worker.
- INT8 requires CUDA compute capability 8.0 or newer because the MoE path has
  that requirement.
- The Qwen3 YAML remains on FP8 by default and documents the available
  `fp8`, `int8`, and `fp16` values.

Configuration precedence clarification:

- Running `bash examples/qwen3-8B-rlvr_megatron/run_rlvr_pipeline.sh` with no
  override uses the YAML's explicit `quantization: fp8`; it is not FP16.
- Passing `actor_infer.strategy_args.strategy_config.quantization=fp16`
  selects FP16 by removing the vLLM quantization backend and forcing the actor
  inference dtype to `float16`.
- Removing the `quantization` key, or overriding it with `null`, selects no
  quantization backend. In the current YAML, `actor_infer.model_args.dtype` is
  `bf16`, so that unquantized mode is BF16.

Verification:

```text
FP16/float16 alias normalization: passed
INT8 backend registration and vLLM ModelConfig parsing: passed
per-channel quantization round-trip/error-bound and zero-row checks: passed
online INT8 weight-loader layout/scale update: passed
CUTLASS parameter layout and dynamic-activation call contract: passed
Hydra composition with fp16 and int8 overrides: passed
24 passed, 6 warnings (new quantization tests plus scheduler/functionals/dynamic batching regressions)
py_compile: passed
git diff --check: passed for the new/modified quantization source and tests
```

Attempting to build the complete vLLM engine config in this managed container
stopped before engine creation with:

```text
RuntimeError: Failed to infer device type
```

The container exposes neither CUDA nor `nvidia-smi`; model-only vLLM config
creation did accept `roll_int8` and resolve `RollInt8Config`. Therefore the
remaining pending step is a live GPU smoke test for both modes, including one
Megatron-to-vLLM model update. Example overrides:

```bash
bash examples/qwen3-8B-rlvr_megatron/run_rlvr_pipeline.sh \
  actor_infer.strategy_args.strategy_config.quantization=fp16

bash examples/qwen3-8B-rlvr_megatron/run_rlvr_pipeline.sh \
  actor_infer.strategy_args.strategy_config.quantization=int8
```

### 14. Five-run quantization experiment matrix

The planned Qwen3-1.7B math comparison keeps the dataset, seed, and all other
training hyperparameters fixed. The five runs are:

```text
qwen3-1p7b-math-fp8-tis-no-uv-traincheck-s42       fp8,  TIS=true,  UV=false
qwen3-1p7b-math-int8-baseline-no-tis-no-uv-s42     int8, TIS=false, UV=false
qwen3-1p7b-math-int8-ours-tis-uv-s42               int8, TIS=true,  UV=true
qwen3-1p7b-math-fp16-baseline-no-tis-no-uv-s42     fp16,TIS=false, UV=false
qwen3-1p7b-math-fp16-ours-tis-uv-s42               fp16,TIS=true,  UV=true
```

Each command must override `exp_name`, `output_dir`, `logging_dir`,
`checkpoint_config.output_dir`, `tracker_kwargs.log_dir`, and the W&B `name`
so runs do not share result directories and offline W&B runs have deterministic
labels. Hydra composition of all five override sets passed. This validates the
configuration syntax only; none of the five GPU training jobs has been run in
the managed environment.

### 15. INT8 CUTLASS weight-stride failure on live H200 startup

Observed on 2026-07-26 while starting the INT8 Qwen3-1.7B experiment on the
H200 GPU container:

```text
RuntimeError: Expected b.stride(0) == 1 to be true, but got false.
```

The exception came from `torch.ops._C.cutlass_scaled_mm.default` during the
vLLM V1 engine's startup `profile_run()`. The later `ActorDiedError` was a
consequence of the vLLM engine-core process exiting; this was not an OOM. The
live environment remains Python 3.10.20, Torch 2.6.0+cu124, vLLM 0.8.4, and
Ray 2.47.1.

Confirmed cause:

- The custom INT8 method created its compute weight directly as a normal
  contiguous `[input, output]` tensor, whose strides are `(output, 1)`.
- vLLM's CUTLASS scaled-matmul contract requires the B operand to be a
  column-major `[input, output]` view with strides `(1, input)`. vLLM's own
  W8A8 CUTLASS backend loads `[output, input]` and transposes it after loading.
- Existing CPU tests asserted only weight shape/dtype, so they did not catch
  the invalid memory layout before the live CUDA call.

Changes made:

- Allocate the INT8 parameter as contiguous `[output, input]` storage exposed
  through its transposed `[input, output]` view. This gives CUTLASS
  `weight.stride(0) == 1` without changing the logical matrix shape.
- Keep online loading through the contiguous `[output, input]` backing view,
  so quantized Megatron updates preserve the CUTLASS layout.
- Extend tests to assert the exact stride both after parameter creation and
  after an online weight-loader update; the mocked CUTLASS call now rejects a
  non-column-major B operand.
- Confirmed every dense Qwen3-1.7B projection used here (QKV, output,
  gate/up, and down) has K and N dimensions divisible by CUTLASS's required 16.

Verification in the managed non-GPU environment:

```text
10/10 quantization tests passed
17 passed, 2 warnings across quantization, functionals, and dynamic batching
manual 128x64 parameter check: stride (1, 128) before and after online load
py_compile/compileall: passed
git diff --check on INT8 source/tests: passed
```

The scheduler-padding regression could not be collected in this managed
environment because import-time `get_node_ip()` attempted a prohibited UDP
socket and raised `PermissionError: [Errno 1] Operation not permitted`; this is
the already-recorded sandbox restriction. The corrected INT8 kernel has not
yet been observed on a live GPU, so the fix remains pending end-to-end
confirmation. Restart the failed container/Ray runtime and rerun the same INT8
command; confirm that vLLM passes `profile_run`, completes a first rollout, an
online model update, and a second rollout.

### 16. GePPO code training and LiveCodeBench dataset adapter

Requested source launcher:

```text
/mnt/shared-storage-user/liujinyi/lj/geppocode/scripts/train_tinyzero_geppo.sh
```

The active GePPO data sources in that launcher are:

```text
train: /mnt/shared-storage-user/liujinyi/lj/geppocode/data/skywork-code/train/train_1p5b_code.pkl
validation: /mnt/shared-storage-user/liujinyi/lj/geppocode/livecodebench/livecodebench_2408_2502.parquet
LiveCodeBench caches: /mnt/shared-storage-user/liujinyi/lj/geppocode/livecodebench/livecodebench_2408_2502/*.pkl
```

The four additional DeepScaler validation paths named by the GePPO shell script
are absent locally, and their loaders/evaluation are commented out in GePPO's
trainer. They were therefore not added to the ROLL launcher.

Relevant verified environment versions:

- Python `3.10.20`
- `datasets==3.1.0`
- `pyarrow==25.0.0`
- `transformers==4.51.1`
- `torch==2.6.0+cu124`
- `pyext` is not installed in the current `roll-qwen3-rlvr` environment

Observed adaptation failures during full-data verification:

```text
ValueError: GePPO sample '11' does not contain input/output tests
```

Cause: the 5314-row training pickle contains three test schemas, not just
stdin/stdout: 3261 plain input/output rows, 912 function-call input/output rows,
and 1141 LeetCode `entry_point/import_prefix/test_code` rows. The latter must
use ROLL's `check_based` evaluator.

The first implementation also expanded every LiveCodeBench cache into the
Arrow validation dataset. The cache directory is 3.4 GiB, and the full load was
terminated with exit code 137:

```text
/bin/bash: line 12: 4 Killed ...
```

Cause: LiveCodeBench parquet stores only `question_id`; its public/private tests
live in 279 per-question pickles, some as large as about 189 MiB. Eagerly
materializing all hidden tests duplicates several GiB in memory.

A direct local evaluator import initially also failed with:

```text
ModuleNotFoundError: No module named 'pyext'
```

Cause: `roll/utils/local_code/testing_util.py` imported `pyext.RuntimeModule`
unconditionally, while the active vLLM requirements/environment do not include
`pyext`.

Changes made:

- Added the registered `geppo_code` dataset adapter and `DataArguments` support
  for a LiveCodeBench cache directory.
- Normalized GePPO chat prompts and all three training-test schemas into ROLL's
  `messages`, `case_type`, `test_cases`, `test_case_function`, and
  `tag=geppo_code` contract.
- Kept LiveCodeBench validation rows lightweight. They carry only a trusted
  local cache descriptor; the code reward worker loads the current question's
  tests lazily and reuses that load for adjacent sampled responses.
- Added a local unpickler mapping for GePPO's LiveCodeBench dataclasses, so ROLL
  does not need GePPO on `PYTHONPATH`. As with the source workflow, these pickle
  files must be trusted local input.
- Added a standard-library `ModuleType` execution fallback when `pyext` is not
  installed.
- Corrected validation preprocessing to use validation's own template and data
  arguments rather than the training data arguments.
- Added `run_geppo_code_pipeline.sh`, which composes the existing Qwen3-1.7B
  config into a single `code_sandbox` domain and names the default run
  `qwen3-1p7b-geppo-code-fp8-tis-uv-s42`.

Run command:

```bash
bash examples/qwen3-8B-rlvr_megatron/run_geppo_code_pipeline.sh
```

The launcher keeps caller overrides last. For example, choose INT8 and a
matching experiment name with:

```bash
GEPPO_EXP_NAME=qwen3-1p7b-geppo-code-int8-tis-uv-s42 \
bash examples/qwen3-8B-rlvr_megatron/run_geppo_code_pipeline.sh \
  actor_infer.strategy_args.strategy_config.quantization=int8
```

Verification results:

```text
training load: 5314/5314 rows
training case types: input=4173, check_based=1141
training function-call rows: 912
training rows with empty tests: 0
validation load: 279/279 rows, 0 missing cache descriptors
1600-token filter: train 5309/5314, validation 278/279
real abc370_a stdin execution-core check: 4/4 tests passed
real 3676 function-call execution-core check: 34/34 tests passed
Hydra + RLVRConfig: code_sandbox-only domain, geppo_code tag routing passed
quantization=fp8, TIS=true, update_validation=true matched the default run name
10 tests passed: GePPO adapter + functionals + dynamic batching
py_compile, bash -n, and diff whitespace checks passed
```

Primary verification commands used from the repository root:

```bash
HF_HOME=/tmp/roll-hf-final-tests PYTHONPATH="$PWD" \
  /root/miniconda3/envs/roll-qwen3-rlvr/bin/python -m pytest \
  tests/datasets/test_geppo_code.py tests/utils/test_functionals.py \
  tests/utils/test_dynamic_batching.py -q

/root/miniconda3/envs/roll-qwen3-rlvr/bin/python -m py_compile \
  roll/configs/data_args.py roll/datasets/dataset.py \
  roll/datasets/geppo_code.py \
  roll/pipeline/rlvr/rewards/code_sandbox_reward_worker.py \
  roll/pipeline/rlvr/rlvr_pipeline.py \
  roll/utils/local_code/testing_util.py

bash -n examples/qwen3-8B-rlvr_megatron/run_geppo_code_pipeline.sh
```

The complete `CodeSandboxRewardWorker` multiprocessing path could not execute in
the managed shell because both import-time node-IP discovery and Python's
`multiprocessing.Manager` require sockets, which the sandbox rejects with
`PermissionError: [Errno 1] Operation not permitted`. This is the existing
managed-shell restriction. A GPU/Ray training step has not been observed, so
end-to-end status remains pending. Next step: run the launcher in the GPU
container, confirm dataset preprocessing reports 5309 train and 278 validation
rows, then observe one rollout/reward/train step and one validation reward.

### 17. GePPO versus ROLL code-reward parity audit

Question: whether the adapted ROLL reward is exactly identical to the reward
used by `geppocode/scripts/train_tinyzero_geppo.sh`.

Confirmed aligned behavior:

- The launcher uses GePPO's `compute_score` with its default
  `is_binary_reward=True`; ROLL also emits one binary 0/1 score and requires all
  tests to pass.
- `</think>` content removal, stdin cases, function-call cases,
  `check(candidate)` cases, and the complete LiveCodeBench public/private tests
  are routed to the corresponding ROLL test mode.
- Known-correct real samples passed in ROLL: `abc370_a` 4/4 stdin tests and
  question `3676` 34/34 function tests.

Confirmed non-parity that can change a sample's binary reward:

- For TACO/LeetCode rows, GePPO accepts only the last exact
  `````python\n...````` block. ROLL's generic extractor accepts raw code and
  usually selects the first block containing a function, otherwise the first
  fenced block.
- For LiveCodeBench, GePPO's extractor selects the last fenced block and returns
  an empty program when no fence exists; ROLL accepts an unfenced program.
- GePPO uses a 6-second test timeout and an effective 2-second outer timeout for
  `check(candidate)` subprocesses. ROLL currently uses 20 seconds for both
  paths.
- GePPO's cached LiveCodeBench scorer uses its own newer
  `lcb_runner/evaluation/testing_util.py`; ROLL uses an older, modified local
  runner. Their timeout, process/memory handling, and some output-comparison
  details are not bit-for-bit identical.

Conclusion: dataset/test coverage and the binary reward meaning align for
ordinary single-code-block, fast deterministic solutions, but exact per-response
GePPO reward parity is **not yet implemented**. This matters for a strict
baseline comparison because raw code, multiple code blocks, borderline timeouts,
or runner edge cases can receive different rewards.

No reward-parity code change was made during this audit. The pending robust fix
is to add an explicit `geppo_exact` reward backend that calls the same GePPO
`compute_score` implementation (with its 6/2-second limits and exact extractor),
or vendor and pin that scorer inside ROLL. Do not label the current reward as
exactly reproduced until a response corpus has been scored by both backends and
the per-sample 0/1 arrays match.

### 18. Independent GePPO-compatible code reward backend

This session implements the pending backend from section 17 without changing
ROLL's generic `CodeSandboxRewardWorker`. Relevant environment versions remain:

- Python `3.10.20`
- Torch `2.6.0+cu124`
- `datasets==3.1.0`
- `pyarrow==25.0.0`
- `psutil==7.2.2`
- NumPy `1.26.4`

Observed environment errors during verification:

```text
OSError: [Errno 30] Read-only file system:
'/root/.cache/huggingface/datasets/...lock'

PermissionError: [Errno 1] Operation not permitted
  at multiprocessing.managers.Server -> SocketListener

ModuleNotFoundError: No module named 'anthropic'
```

Confirmed causes:

- Hugging Face `datasets` tried to create generator locks in the managed
  shell's read-only `/root/.cache`; setting `HF_HOME` and
  `HF_DATASETS_CACHE` under `/tmp` fixes test execution.
- GePPO's `multiprocessing.Manager` result transport creates a local socket
  server, which this managed shell prohibits. This transport is not part of
  the scoring semantics.
- Importing GePPO's evaluator through its package executes the package
  `__init__` and transitively imports an optional Anthropic prompt dependency.
  Loading the standalone evaluator source avoids that unrelated dependency.

Changes made:

- Added `roll/utils/geppo_code_reward`, an internal scorer that preserves
  GePPO's exact training and LiveCodeBench code extraction, top-level function
  conversion, stdout/Decimal comparison, function-call execution, reliability
  guard, 6-second per-test timeout, 2-second outer script-test timeout, binary
  reward default, and optional partial/power-four rewards.
- Kept each generated program in a dedicated child process with GePPO's global
  timeout and 50 GiB RSS watchdog. Replaced only the Manager proxy transport
  with a one-way process pipe, removing the local-socket-server requirement
  without changing test results.
- Added `GeppoCodeRewardWorker`, which decodes responses with
  `skip_special_tokens=True`, consumes serialized `geppo_ground_truth`, and
  emits ROLL-compatible float16 token rewards, response rewards, and scores.
- Added serialized `geppo_ground_truth` to every adapter row. This avoids Arrow
  union-schema `None` keys changing GePPO's key-presence dispatch when training
  data mixes stdin/function-call and `check(candidate)` rows.
- Added GePPO reward fields to `RewardConfig` and changed the dedicated GePPO
  launcher to select the new worker with timeout 6, binary rewards enabled,
  the per-question LiveCodeBench cache directory, and an explicit FP8 override
  matching its default experiment name. Caller overrides remain last, so the
  same launcher can still select INT8 or FP16.
- Added focused adapter and reward tests for fence requirements, last-block
  selection, function wrapping, binary/partial/power-four rewards,
  `check(candidate)`, isolated stdin execution, and lazy LiveCodeBench caches.

Observed verification results:

```text
py_compile/compileall: passed
bash -n run_geppo_code_pipeline.sh: passed
11 focused GePPO adapter/reward tests: passed
real training dataset: 5314 rows
  stdin rows: 3261
  function-call rows: 912
  check(candidate) rows: 1141
real LiveCodeBench validation dataset: 279 rows
ROLL evaluator abc370_a: 4/4 passed
GePPO evaluator abc370_a: 4/4 passed
ROLL evaluator 3676: 34/34 passed
GePPO evaluator 3676: 34/34 passed
real training row 11 check(candidate): 2/2 passed
full ROLL LiveCodeBench scoring:
  correct fenced stdin=True, raw stdin=False, wrong fenced stdin=False
  correct fenced function=True
Hydra + RLVRConfig:
  worker=GeppoCodeRewardWorker, dataset=geppo_code,
  tag routing geppo_code->code_sandbox, timeout=6, binary=True,
  default launcher quantization=fp8
caller override after launcher default: quantization=int8
direct worker contract check:
  token rewards shape=(1, 3), response reward=[1.0], score=[1.0],
  output dtype=torch.float16
```

Primary test command:

```bash
HF_HOME=/tmp/roll-hf-cache \
HF_DATASETS_CACHE=/tmp/roll-hf-cache/datasets \
PYTHONPATH="$PWD" \
/root/miniconda3/envs/roll-qwen3-rlvr/bin/python -m pytest -q \
  tests/utils/test_geppo_code_reward.py tests/datasets/test_geppo_code.py
```

The scorer, real datasets, real evaluation cores, full isolated-process score,
and final Hydra configuration are observed working. A complete GPU/Ray rollout
and reward step has not been run in this managed shell because ROLL's import-time
node-IP discovery opens a prohibited UDP socket. Next pending step: launch
`run_geppo_code_pipeline.sh` in the normal GPU container and observe one rollout,
one reward computation, one actor update, and one validation reward before
calling the whole training pipeline end-to-end confirmed.

### 19. Coding update-validation flattened `test_cases`

Observed on 2026-07-27 in the live coding run:

```text
AssertionError: key test_cases length 8008 is not equal to batch size 256
```

The traceback occurred in `DataProto.concat(all_outputs)` during
`RLVRPipeline._generate_validation_batch`, after generation and before the
validation reward call. Relevant versions remain Python `3.10.20`, Torch
`2.6.0+cu124`, NumPy `1.26.4`, and Ray `2.47.1`.

Confirmed cause:

- Update validation expanded every prompt's metadata with
  `np.repeat([v[0]], len(output_tokens))` and did not pass an axis.
- For scalar strings and dictionaries this appeared to work, but
  `test_cases` is a nested list. NumPy inferred its inner test-case dimension,
  flattened it, and repeated every individual test instead of treating the
  complete list as one response-level metadata value.
- The generated tensor batch correctly contained 32 prompts x 8 responses =
  256 rows, while the flattened test lists produced 8008 top-level metadata
  rows, so `DataProto.check_consistency()` correctly rejected the mismatch.
- The intermediate `single_prompt.non_tensor_batch` also used plain Python
  lists instead of the protocol's required one-dimensional `dtype=object`
  arrays, making NumPy shape inference easier to trigger.

Changes made:

- Added `make_non_tensor_array`, which always creates a one-dimensional object
  array and never infers axes from nested list/array values.
- Added `repeat_non_tensor_value`, which repeats a complete metadata object as
  one value per response without flattening it.
- Updated fixed/random/rollout update-validation metadata preparation to use
  the safe one-dimensional construction path.
- Updated per-prompt and per-response validation expansion to use the safe
  repeat helper instead of `np.repeat([nested_value], n)`.
- Added a regression with 32 prompts, 8 responses per prompt, and 31 nested
  test cases per response. It asserts a 256-row tensor batch and exactly 256
  top-level `test_cases` objects; the previous implementation would create
  7936 top-level objects in that fixture.

Observed verification:

```text
py_compile: protocol, RLVR pipeline, and protocol regression passed
38 tests passed, 2 environment-version warnings:
  protocol, GePPO reward, GePPO dataset, dynamic batching, functionals
real GePPO training metadata smoke test:
  prompt_count=32
  response_batch=256
  test_cases_top_level=256
  objects_preserved=True
git diff --check on the modified tracked files: passed
```

The exact failing GPU/Ray validation step cannot run in this managed shell.
The next pending step is to restart the coding launcher and observe update
validation pass `DataProto.concat`, compute its GePPO rewards, and finish the
rollback decision. Do not mark the live coding run fixed until that result is
observed.

### 20. Coding baseline launcher stopped before reward overrides

Observed on 2026-07-27 after the user edited
`run_geppo_code_pipeline.sh`:

```text
AssertionError: domain dataset code_sandbox has no data
```

Confirmed cause:

- A blank physical line was inserted immediately after the continued
  `+actor_infer.strategy_args.strategy_config.quantization=fp8 \` line.
- Bash consumed the backslash-newline pair, then the blank line terminated the
  `run_rlvr_pipeline.sh` command. The following `~rewards.math_rule` and
  `+rewards.code_sandbox=...` lines were therefore not Hydra arguments.
- Overrides before the blank line did remove the `math_rule` sampling ratio and
  add `domain_interleave_probs.code_sandbox=1.0`, while the effective reward
  config still contained only `math_rule`.
- The effective tag map was consequently only
  `deepmath_103k/aime -> math_rule`. The GePPO adapter emits
  `tag=geppo_code`, so `update_dataset_domain` used its `math_rule` fallback;
  filtering for the requested `code_sandbox` domain then produced zero rows and
  the assertion correctly failed.

Observed configuration before the shell break:

```text
domain_interleave_probs={'code_sandbox': 1.0}
reward_domains=['math_rule']
tag_2_domain={'deepmath_103k': 'math_rule', 'aime': 'math_rule'}
geppo_code maps to=math_rule fallback
```

Required launcher correction: delete the blank line between the FP8 override
and `~rewards.math_rule`, keeping every continued argument on adjacent physical
lines. No Python source change is required. `bash -n` still passes the broken
script because this is valid Bash syntax with the wrong command boundary, not a
syntax error.

Hydra composition with the same overrides treated as one command was verified:

```text
domain_interleave_probs={'code_sandbox': 1.0}
reward_domains=['code_sandbox']
tag_2_domain={'geppo_code': 'code_sandbox'}
train_infer_ratio_trunc_is=False
update_validation.enable=False
```

The user-owned shell change was diagnosed but not modified during this session.
Next pending step: remove that one blank line, rerun the launcher, and observe a
non-empty `code_sandbox` dataset. Do not mark the live run fixed until startup
passes the assertion.

### 21. Math validation tokenizer received an empty batch

Observed on 2026-07-27 when launching the Qwen3-1.7B math INT8/TIS/update-
validation experiment:

```text
IndexError: list index out of range
  File "roll/pipeline/rlvr/rlvr_pipeline.py", line 105, in encode_function
    encodings = tokenizer(text_list)
  File "transformers/tokenization_utils_fast.py", line 572, in _batch_encode_plus
    for key in tokens_and_encodings[0][0].keys():
```

Relevant dependency versions are Python `3.10.20`, Torch `2.6.0+cu124`,
NumPy `1.26.4`, Ray `2.47.1`, datasets `3.1.0`, and Transformers `4.51.1`.

Confirmed cause:

- The composed command leaves both `validation.data_args.messages` and
  `validation.data_args.prompt` as `None`.
- The validation JSONL has standard `messages` and `prompt` columns, but the
  old expressions `getattr(data_args, "messages", "messages")` and
  `getattr(data_args, "prompt", "prompt")` return `None` because those
  dataclass attributes exist. The fallback strings only apply when an
  attribute is absent, not when its value is `None`.
- Neither `None` key matched the dataset batch, leaving `text_list=[]`; the
  Transformers fast tokenizer then indexed its empty encoded result.
- This is an upstream latent bug present in the Git baseline since the column
  selection was introduced. The GePPO/coding adapter did not change the
  `prompt` or `messages` defaults. Its launcher explicitly supplies
  `validation.data_args.messages=messages`, which masks the old bug, while the
  direct math launcher does not.
- The failure occurs during CPU dataset preprocessing before Ray/GPU worker
  initialization, so INT8, TIS, and update-validation math are not causal.

Changes made:

- Updated `get_encode_function` to honor an explicitly configured `messages`
  or `prompt` column first, otherwise auto-detect the standard `messages` and
  `prompt` columns.
- Added actionable `ValueError` messages when an explicit column is absent or
  neither standard column exists, instead of passing an unexplained empty list
  to the tokenizer.
- Added focused tests for automatic `messages` selection, `prompt` fallback,
  explicit prompt precedence, and a missing explicit column.

Observed verification:

```text
exact Hydra overrides:
  validation.messages=None
  validation.prompt=None
  dataset columns include messages and prompt
  old resolved keys=None/None and text_list length=0

real Qwen3-1.7B tokenizer + math_benchmarks_0710_newest.jsonl:
  source rows=2032
  all 2032 rows encoded without tokenizer errors
  rows after prompt_length=512 filter=2027
  first encoded prompt length=163

42 tests passed:
  RLVR dataset encoding, protocol nested metadata, GePPO reward/dataset,
  dynamic batching, and functionals
```

The exact preprocessing crash is fixed and verified against the real model
tokenizer and validation file. A full Ray/GPU training startup cannot run in
this managed shell; the next pending step is to rerun the original experiment
command in the GPU container and observe worker initialization and at least one
training/update-validation step. Do not mark the complete training run as
confirmed until that result is observed.

### 8. W&B offline sync helper

Observed issue while adding a sync helper:

```text
Found 1 wandb offline run(s) under /mnt/shared-storage-user/liujinyi/lj/rlvr_runs
[dry-run] wandb sync $'...offline-run-20260727_010400-utxfjm5s\n...'
```

Confirmed cause:

- The discovery pipeline printed run directories with newline separators and
  then fed that stream into `sort -zu`, so all discovered directories collapsed
  into a single NUL-delimited record.
- The fix is to emit `\0` from the `dirname` loop before sorting and
  deduplication.

Environment facts:

- `bash==5.1.16(1)-release`
- `wandb` CLI is not available in this shell's `PATH`

Changes made:

- Added `scripts/sync_wandb_runs.sh`.
- The script scans a configurable root for `*.wandb` and `*.wnb` files,
  deduplicates the offline run directories, and runs `wandb sync` with bounded
  concurrency.
- Supported options: `--root`, `-j/--jobs`, `-i/--interval`, `--watch`, and
  `--dry-run`.
- Fixed the discovery stage to output NUL-delimited run directories so each
  offline run is handled separately.

Observed verification:

```text
bash -n scripts/sync_wandb_runs.sh
scripts/sync_wandb_runs.sh --dry-run --root /mnt/shared-storage-user/liujinyi/lj/rlvr_runs

Found 15 wandb offline run(s) under /mnt/shared-storage-user/liujinyi/lj/rlvr_runs
[dry-run] wandb sync /mnt/shared-storage-user/liujinyi/lj/rlvr_runs/qwen3-1p7b-math-fp16-baseline-no-tis-no-uv-s42/wandb/wandb/offline-run-20260726_184811-i1wholwx
...
```

Next pending step:

- Run the helper in an environment that actually has the `wandb` CLI installed
  and confirm a real sync succeeds.

### 9. Phi-2 math FP8 baseline launcher

Requested launcher:

- Model: `/mnt/shared-storage-user/liujinyi/models_hf/phi-2`
- Task: math RLVR
- Quantization: FP8 for vLLM actor inference
- TIS: disabled
- Update validation: disabled

Confirmed facts:

- The phi-2 model directory exists.
- Required local files observed:
  - `config.json`
  - `tokenizer.json`
  - `model.safetensors.index.json`
- `config.json` has `model_type: "phi"` and `max_position_embeddings: 2048`.

Changes made:

- Added `examples/qwen3-8B-rlvr_megatron/run_phi2_math_fp8_baseline.sh`.
- The script writes outputs under
  `/mnt/shared-storage-user/liujinyi/lj/rlvr_runs/${EXP_NAME}` by default.
- Default experiment name:
  `phi2-math-fp8-baseline-no-tis-no-uv-s42`.
- The script sets:
  - `pretrain=/mnt/shared-storage-user/liujinyi/models_hf/phi-2`
  - `reward_pretrain=/mnt/shared-storage-user/liujinyi/models_hf/phi-2`
  - `train_infer_ratio_trunc_is=false`
  - `update_validation.enable=false`
  - `+actor_infer.strategy_args.strategy_config.quantization=fp8`
  - `prompt_length=512`
  - `response_length=1536`
  - train/infer/reference/reward templates to `base`

Rationale:

- Phi-2 is not a Qwen chat model, so the launcher uses ROLL's `base` prompt
  template instead of `qwen2_5/native`, which relies on tokenizer chat-template
  support.
- `prompt_length + response_length` is capped at 2048 to match phi-2's context
  length.

Observed verification:

```text
bash -n examples/qwen3-8B-rlvr_megatron/run_phi2_math_fp8_baseline.sh
model_dir_exists True
config.json True
tokenizer.json True
model.safetensors.index.json True
```

Hydra composition was attempted but not completed in this managed shell because
the shell's Python does not have Hydra installed:

```text
ModuleNotFoundError: No module named 'hydra'
```

Next pending step:

- Run the launcher inside the real `roll-qwen3-rlvr` environment and observe
  Hydra composition plus Ray/GPU worker initialization. Do not mark phi-2
  training as verified until the run reaches at least worker startup.

Follow-up finding:

- The phi-2 launcher now prints its experiment name, model path, output root,
  quantization mode, and TIS/update-validation switches before delegating to
  `run_rlvr_pipeline.sh`, so future runs should not look completely silent.
- Current ROLL can likely use phi-2 through vLLM/Hugging Face inference paths,
  but the current Megatron RLVR path is not confirmed compatible with phi-2.
- `mcore_adapter` registers converter templates for model types such as
  `qwen2`, `qwen3`, `llama`, `mistral`, `mixtral`, etc.; no `phi` registration
  was found.
- `AutoConfig`/`AutoModel` fall back to generic `McaModelConfig`/`McaGPTModel`
  when a model type is unregistered, but converter template lookup uses
  `templates[name]` directly. With `model_type: "phi"`, that is expected to
  fail during HF-to-Megatron conversion unless a phi template is added or the
  run is moved to a Hugging Face/DeepSpeed training strategy.

Next pending step:

- For actual phi-2 RLVR training, either add/register a phi converter template
  for Megatron or make a separate DeepSpeed/HF-based phi-2 config. The existing
  Megatron launcher should be treated as a smoke-test attempt, not a confirmed
  phi-2-compatible training path.

### 22. Qwen3-1.7B INT8 resume launcher with TIS and no update validation

Requested source run:

```text
/mnt/shared-storage-user/liujinyi/lj/rlvr_runs/qwen3-1p7b-math-int8-tis-uvtestthre-s42
```

Confirmed facts:

- The source run currently contains a complete `checkpoint-100` under the
  `20260726-190634` checkpoint session, including pipeline state.
- The original W&B metadata used INT8 inference, TIS, update validation, and
  no template command-line overrides.
- The effective template defaults are supplied by `rlvr_config.yaml`; they are
  not uniform across workers, so forcing one template would not preserve the
  requested defaults.

Changes made:

- Added
  `examples/qwen3-8B-rlvr_megatron/resume_qwen3_1p7b_math_int8_tis_no_uv.sh`.
- The launcher selects the newest checkpoint unless `RESUME_CHECKPOINT` is
  provided, preserves config-default templates, enables TIS, explicitly
  disables update validation, and retains INT8 actor inference.

Observed verification:

```text
bash -n: passed
auto-selected checkpoint: 20260726-190634/checkpoint-100
Hydra composition:
  train_infer_ratio_trunc_is=True
  update_validation.enable=False
  actor inference quantization=int8
  actor train/infer, reference, validation templates=qwen2_5
  math reward template=qwen3
git diff --check: passed
```

The full application `--cfg job` path cannot run in this managed shell because
import-time logging attempts to write under the read-only repository and socket
creation is denied with `PermissionError: [Errno 1] Operation not permitted`.
Pure Hydra composition succeeded without importing the Ray/GPU application.
The next pending step is to run the launcher in the GPU environment and observe
it restore `checkpoint-100` and proceed to step 101 or later.

## Maintenance Rules

- Record exact errors and versions, not guesses.
- Mark suggested fixes as pending until their result is observed.
- Update statuses instead of duplicating entries.
- Keep commands runnable from the repository root.
- Preserve unrelated user changes.

### 11. SGLang v0.4.6.post4 uvloop event-loop setup failure

Observed error from the user's `sglang` Conda environment while launching
`run_qwen3_1p7b_math_sglang_tis_uv_min800_4gpus.sh` on 2026-08-01:

```text
ray.exceptions.RayTaskError(RuntimeError): ray::ActorWorker.setup_collective_group()
  File "roll/third_party/sglang/v046post4_patch/engine.py", line 28, in setup_collective_group
    loop = asyncio.get_event_loop()
  File "/root/miniconda3/envs/sglang/lib/python3.10/site-packages/uvloop/__init__.py", line 206, in get_event_loop
    raise RuntimeError(
RuntimeError: There is no current event loop in thread 'MainThread'.
```

Relevant dependency/config facts from the traceback and repository requirements:

- Python path in traceback: `/root/miniconda3/envs/sglang/lib/python3.10`.
- Active ROLL SGLang patch selected by the traceback:
  `roll/third_party/sglang/v046post4_patch`, matching
  `sglang[srt,torch-memory-saver]==0.4.6.post4` in
  `requirements_torch260_sglang.txt`.
- The failing event-loop policy is uvloop, from the traceback frame in
  `uvloop/__init__.py`.
- The launcher is using `SGLANG_QUANTIZATION=fp8`, `SGLANG_TP_SIZE=1`,
  `MEGATRON_TRANSFORMER_IMPL=transformer_engine`, and
  `GENERATE_OPT_LEVEL=0`.

Confirmed cause: ROLL's SGLang engine patch used `asyncio.get_event_loop()` in
synchronous model-update methods. Under the current uvloop/Python 3.10 runtime,
Ray actor `MainThread` has no pre-created current loop at the time
`actor_infer-0` executes `setup_collective_group`, so `get_event_loop()` raises
before the collective group is created. This is not an SGLang model/env install
failure; it is missing event-loop initialization in the ROLL SGLang patch.

Code change applied:

- `roll/third_party/sglang/v046post4_patch/engine.py`: added
  `_ensure_event_loop()` that creates and installs a new loop when
  `asyncio.get_event_loop()` raises `RuntimeError`; all synchronous SGLang
  model-update entrypoints now use that helper before `run_until_complete()`.

Verification performed:

```text
PYTHONPYCACHEPREFIX=/tmp/roll_pycache python -m py_compile roll/third_party/sglang/v046post4_patch/engine.py
py_compile exit=0
```

A direct import/runtime helper check in this Codex shell could not be completed
because this shell's `python` could not import `sglang`. Runtime observation from
the user's rerun did, however, show all `actor_infer` workers start their SGLang
async services and reach a SGLang prefill request. That occurs after
`setup_collective_group`, so the event-loop fix is confirmed to pass the
original failure point. Any subsequent traceback should be recorded separately.

### 12. SGLang update-validation async request does not complete

Initial observation after applying the event-loop fix on 2026-08-01:

```text
(ActorWorker(actor_infer-2) ...) Prefill batch. #new-seq: 1,
#new-token: 130, #cached-token: 0, token usage: 0.00, #running-req: 0,
#queue-req: 0
```

Confirmed configuration/path facts:

- `rlvr_config.yaml` sets `eval_steps: 25`, `response_length: 8192`, and
  `validation.generating_args.max_new_tokens: ${response_length}`.
- The validation file `data/math_benchmarks_0710_newest.jsonl` has 2032 rows.
- `RLVRPipeline.run()` runs validation whenever `global_step % eval_steps == 0`,
  which includes the first iteration at `global_step == 0`.
- `RLVRPipeline.val()` calls the validation scheduler with
  `batch_size=len(self.val_dataset)`, so the initial evaluation requests all
  2032 prompts before regular rollout and training begin.

The later driver and worker logs correct the initial interpretation: ordinary
validation completed successfully at `14:32:59` and the first ordinary rollout
completed at `14:33:55` (`prepare_cached_rollout: cached batch_size=512`). The
driver then saved the guard state at `14:34:39`, and at `14:36:06` entered
`_generate_validation_batch`, logging its sampled UV prompt fields. All four
`actor_infer` workers then logged `generate server global step 0` and
`[sglang asyncio] env setup (done)`. `actor_infer-2` subsequently logged the
first SGLang prefill, but no request-completion callback or later driver log
appeared.

Confirmed scope: this is the post-training `update_validation` path, not the
regular `validation:` benchmark path and not the event-loop initialization
fix. `RLVRPipeline._generate_validation_batch()` unconditionally starts the
asynchronous server and, in legacy mode, calls the blocking
`ActorWorker.generate_request()` path. Each such request waits up to 600
seconds for the callback. The current UV configuration uses 32 prompts, 8
responses per prompt, and `max_new_tokens=8192`; the SGLang stream for the
first request is not yielding a terminal callback after prefill.

The run directory name contains `nouv`, but the runtime proof above (guard
sync plus `Update Validation:` log) means `update_validation.enable` was true
in that run. This should be treated as a configuration/UV-path issue, not as a
successful no-UV run.

Status: the exact SGLang 0.4.6.post4 stream/callback compatibility cause is
not yet verified. First isolate it by rerunning with
`update_validation.enable=false`; ordinary validation, ordinary rollout, and
the train/update path have all progressed past the relevant points in the
observed run. For a UV smoke test, reduce the UV request to one prompt, one
sample, and a short response before restoring the 32 x 8 x 8192 workload.

### 13. SGLang update-validation metrics-only synchronous path

Goal: collect update-validation (UV) metrics with SGLang without rollback. The
user does not require rollback or update-norm metrics for this run.

Confirmed cause and scope:

- The normal SGLang `SgLangStrategy.model.generate()` path completed ordinary
  rollout generation in the affected run.
- The stuck path is specifically ROLL's UV async lifecycle:
  `start_server -> async_engine -> tokenizer_manager.generate_request(stream=True)
  -> callback`. The first prefill is logged but the callback does not complete.
- This is independent of the earlier missing uvloop event-loop issue. That
  issue was passed after the `EngineSA` event-loop fix.

Code changes applied:

- `roll/distributed/strategy/sglang_strategy.py` now provides
  `SgLangStrategy.generate_with_logprobs()`. It uses synchronous SGLang Engine
  generation with `return_logprob=True` and returns response token IDs aligned
  one-for-one with their SGLang output token logprobs.
- `roll/pipeline/base_worker.py` adds
  `ActorWorker.generate_request_sync()`, a Ray actor method that exposes that
  synchronous SGLang result in the existing UV callback-data format.
- `roll/pipeline/rlvr/rlvr_pipeline.py` selects that synchronous method only
  when `actor_infer.strategy_args.strategy_name=sglang`; it does not call
  `actor_infer.start_server()` or `stop_server()` for SGLang UV. vLLM and other
  strategies retain their legacy/scheduler paths.
- `UpdateValidationConfig` has `metrics_only: bool = False`. With it enabled,
  UV still syncs the post-update actor to inference, generates the validation
  batch, and records advantage metrics, but always accepts the update and does
  not call rollback or LR adaptation.
- `run_qwen3_1p7b_math_sglang_tis_uv_min800_4gpus.sh` exposes
  `UV_ENABLE` (default `false`), `UV_METRICS_ONLY` (default `true`), and
  `UV_LOG_UPDATE_NORM` (default `false`). The last setting prevents the
  otherwise-default guard-model/optimizer copy, which metrics-only UV does not
  need. Set it to `true` only when update-norm metrics are explicitly wanted.

Static verification performed:

```text
py_compile: passed (SGLang strategy, worker, config, and pipeline)
bash -n: passed (SGLang UV launcher)
git diff --check: passed
UV SGLang metrics-only AST contract: OK
```

All static checks passed. A direct SGLang import/runtime test cannot run in the
managed Codex shell because this shell lacks the user's SGLang Python package;
therefore do not mark this path runtime-verified until a GPU run completes a UV
step.

Required first runtime command from the user's SGLang environment:

```bash
ray stop --force
UV_ENABLE=true UV_METRICS_ONLY=true UV_LOG_UPDATE_NORM=false EXP_NAME=qwen3-1p7b-math-sglang-tis-uv-metrics-4g-s42-0801 SGLANG_MEM_FRACTION_STATIC=0.6 SGLANG_TP_SIZE=1 SGLANG_QUANTIZATION=fp8 MEGATRON_TRANSFORMER_IMPL=transformer_engine GENERATE_OPT_LEVEL=0 bash examples/qwen3-8B-rlvr_megatron/run_qwen3_1p7b_math_sglang_tis_uv_min800_4gpus.sh
```

Expected proof in the driver log after the actor update:

```text
SGLang UV synchronous generation completed ...
Update Validation: Generated batch with infer_logprobs=True
UV metrics recorded without rollback or LR adaptation.
```

Next pending step: observe the above lines plus completion of one training
step in the real SGLang GPU environment. If `output_token_logprobs` is absent
there, preserve the new exception and record the exact SGLang response shape;
the normal rollout should still remain unaffected.
