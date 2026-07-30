import json
import os
import pickle
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from datasets import Dataset, load_dataset

from roll.configs.data_args import DataArguments
from roll.datasets.dataset import register_dataset_adapter


_LCB_PICKLE_MODULE = "verl.utils.reward_score.livecodebench.lcb_runner.benchmarks.code_generation"


class _Platform(Enum):
    LEETCODE = "leetcode"
    CODEFORCES = "codeforces"
    ATCODER = "atcoder"


class _Difficulty(Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class _TestType(Enum):
    STDIN = "stdin"
    FUNCTIONAL = "functional"


@dataclass
class _Test:
    input: str
    output: str
    testtype: _TestType


class _CodeGenerationProblem:
    pass


class _LiveCodeBenchUnpickler(pickle.Unpickler):
    """Load trusted GePPO caches without importing GePPO's Python package."""

    _CLASS_MAP = {
        "Platform": _Platform,
        "Difficulty": _Difficulty,
        "TestType": _TestType,
        "Test": _Test,
        "CodeGenerationProblem": _CodeGenerationProblem,
    }

    def find_class(self, module: str, name: str):
        if module == _LCB_PICKLE_MODULE and name in self._CLASS_MAP:
            return self._CLASS_MAP[name]
        return super().find_class(module, name)


def _load_pickle(path: str, *, livecodebench: bool = False) -> Any:
    # Pickle is executable input. This adapter is intentionally limited to
    # user-provided, trusted local datasets and caches.
    with open(path, "rb") as file:
        if livecodebench:
            return _LiveCodeBenchUnpickler(file).load()
        return pickle.load(file)


def _as_string(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def _build_test_cases(inputs: List[Any], outputs: List[Any], row_id: str) -> List[Dict[str, str]]:
    if len(inputs) != len(outputs):
        raise ValueError(
            f"GePPO sample {row_id!r} has {len(inputs)} inputs but {len(outputs)} outputs"
        )
    return [
        {"stdin": _as_string(test_input), "expected_stdout": _as_string(expected_output)}
        for test_input, expected_output in zip(inputs, outputs)
    ]


def _normalize_train_row(row: Dict[str, Any]) -> Dict[str, Any]:
    reward_model = row.get("reward_model") or {}
    ground_truth = reward_model.get("ground_truth") or {}
    extra_info = row.get("extra_info") or {}
    row_id = str(extra_info.get("index", reward_model.get("question_id", "")))
    if not isinstance(row.get("prompt"), list):
        raise ValueError(f"GePPO sample {row_id!r} does not contain a chat-message prompt")

    if {"entry_point", "import_prefix", "test_code"}.issubset(ground_truth):
        case_type = "check_based"
        test_cases = [
            {
                "assert_code": ground_truth["test_code"],
                "entry_point": ground_truth["entry_point"],
                "import_prefix": ground_truth["import_prefix"],
            }
        ]
        test_case_function = ""
    else:
        inputs = ground_truth.get("inputs") or []
        outputs = ground_truth.get("outputs") or []
        if not inputs or not outputs:
            raise ValueError(f"GePPO sample {row_id!r} does not contain supported code tests")
        case_type = "input"
        test_cases = _build_test_cases(inputs, outputs, row_id)
        test_case_function = ground_truth.get("fn_name") or ""

    return {
        "id": row_id,
        "source": row.get("data_source", "geppo_code"),
        "difficulty": extra_info.get("difficulty", ""),
        "messages": row["prompt"],
        "ground_truth": ground_truth,
        "geppo_ground_truth": json.dumps(ground_truth, ensure_ascii=False),
        "case_type": case_type,
        "test_cases": test_cases,
        "test_case_function": test_case_function,
        "tag": "geppo_code",
    }


def _livecodebench_cache_dirs(data_files: List[str], configured_dir: Optional[str]) -> List[Path]:
    if configured_dir:
        return [Path(configured_dir)]
    return [Path(path).with_suffix("") for path in data_files if Path(path).suffix.lower() == ".parquet"]


def _find_livecodebench_cache(question_id: str, cache_dirs: List[Path]) -> Path:
    for cache_dir in cache_dirs:
        candidate = cache_dir / f"{question_id}.pkl"
        if candidate.is_file():
            return candidate
    searched = ", ".join(str(path) for path in cache_dirs) or "<none>"
    raise FileNotFoundError(
        f"LiveCodeBench cache for question {question_id!r} was not found; searched: {searched}"
    )


def _normalize_livecodebench_row(row: Dict[str, Any], cache_dirs: List[Path]) -> Dict[str, Any]:
    reward_model = row.get("reward_model") or {}
    ground_truth = reward_model.get("ground_truth") or {}
    extra_info = row.get("extra_info") or {}
    question_id = str(
        ground_truth.get("question_id")
        or reward_model.get("question_id")
        or extra_info.get("question_id")
        or ""
    )
    if not question_id:
        raise ValueError("LiveCodeBench row does not contain question_id")

    cache_path = _find_livecodebench_cache(question_id, cache_dirs)

    return {
        "id": question_id,
        "source": row.get("data_source", "livecodebench"),
        "difficulty": extra_info.get("difficulty", ""),
        "messages": row["prompt"],
        "ground_truth": {"question_id": question_id},
        "geppo_ground_truth": json.dumps(
            {"question_id": question_id, "cache_path": str(cache_path)},
            ensure_ascii=False,
        ),
        "case_type": "livecodebench",
        # Expanding all hidden tests here can consume several GB. Reward
        # workers load only the current question when it is evaluated.
        "test_cases": [{"question_id": question_id, "cache_path": str(cache_path)}],
        "test_case_function": "",
        "tag": "geppo_code",
    }


def load_livecodebench_tests(cache_path: str) -> tuple[List[Dict[str, str]], str]:
    problem = load_livecodebench_problem(cache_path)
    tests = list(problem.public_test_cases) + list(problem.private_test_cases)
    question_id = str(getattr(problem, "question_id", Path(cache_path).stem))
    if not tests:
        raise ValueError(f"LiveCodeBench question {question_id!r} has no tests")
    metadata = problem.metadata if isinstance(problem.metadata, dict) else json.loads(problem.metadata)
    return (
        _build_test_cases(
            [test.input for test in tests],
            [test.output for test in tests],
            question_id,
        ),
        metadata.get("func_name") or "",
    )


def load_livecodebench_problem(cache_path: str) -> Any:
    return _load_pickle(cache_path, livecodebench=True)


def _iter_source_rows(path: str) -> Iterable[Dict[str, Any]]:
    suffix = Path(path).suffix.lower()
    if suffix in {".pkl", ".pickle"}:
        rows = _load_pickle(path)
        if not isinstance(rows, (list, tuple)):
            raise TypeError(f"Expected a list in GePPO training pickle {path}, got {type(rows).__name__}")
        yield from rows
        return
    if suffix == ".parquet":
        yield from load_dataset("parquet", data_files=[path], split="train")
        return
    raise ValueError(f"Unsupported GePPO dataset file: {path}")


def _generate_rows(
    data_files: List[str],
    livecodebench_data_dir: Optional[str],
    source_fingerprints: Optional[List[tuple[str, int, int]]] = None,
):
    # This argument participates in Hugging Face's generator fingerprint so a
    # replaced source file at the same path does not reuse stale Arrow data.
    del source_fingerprints
    cache_dirs = _livecodebench_cache_dirs(data_files, livecodebench_data_dir)
    for path in data_files:
        for row in _iter_source_rows(path):
            ground_truth = (row.get("reward_model") or {}).get("ground_truth") or {}
            if (
                isinstance(ground_truth, dict)
                and "question_id" in ground_truth
                and "inputs" not in ground_truth
            ):
                yield _normalize_livecodebench_row(row, cache_dirs)
            else:
                yield _normalize_train_row(row)


@register_dataset_adapter("geppo_code")
def geppo_code_dataset(
    data_files: List[str],
    split: str = "train",
    data_args: Optional[DataArguments] = None,
) -> Dataset:
    if split != "train":
        raise ValueError(f"The geppo_code adapter expects split='train', got {split!r}")
    if not data_files:
        raise ValueError("The geppo_code adapter requires at least one local data file")
    missing = [path for path in data_files if not os.path.isfile(path)]
    if missing:
        raise FileNotFoundError(f"GePPO dataset files do not exist: {missing}")

    livecodebench_data_dir = data_args.livecodebench_data_dir if data_args else None
    source_fingerprints = [
        (path, os.path.getsize(path), os.stat(path).st_mtime_ns) for path in data_files
    ]
    return Dataset.from_generator(
        _generate_rows,
        gen_kwargs={
            "data_files": data_files,
            "livecodebench_data_dir": livecodebench_data_dir,
            "source_fingerprints": source_fingerprints,
        },
    )
