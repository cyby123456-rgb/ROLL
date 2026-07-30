# Adapted from GePPO's code-task reward implementation.
import ast
import json
import re
import signal
import subprocess
import tempfile
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from roll.datasets.geppo_code import load_livecodebench_problem
from roll.utils.geppo_code_reward.evaluator import check_correctness


IMPORT_PROMPT = """from typing import *

from functools import *
from collections import *
from itertools import *
from heapq import *
from bisect import *
from string import *
from operator import *
from math import *
import math
import datetime
inf = float('inf')

"""


@contextmanager
def _timeout(seconds):
    def handler(signum, frame):
        del signum, frame
        raise TimeoutError("code execution timed out")

    previous_handler = signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)


def _convert_function_to_class_method(raw_code: str, function_name: str) -> Optional[str]:
    tree = ast.parse(raw_code)
    target = None
    remaining = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            target = node
        else:
            remaining.append(node)
    if target is None:
        return None
    if not (target.args.args and target.args.args[0].arg == "self"):
        target.args.args.insert(0, ast.arg(arg="self", annotation=None))
    remaining.append(
        ast.ClassDef(
            name="Solution",
            bases=[],
            keywords=[],
            body=[target],
            decorator_list=[],
        )
    )
    tree.body = remaining
    return ast.unparse(tree)


def _extract_python_block(solution: str) -> Optional[str]:
    blocks = re.findall(r"```python\n(.*?)```", solution, re.DOTALL)
    return blocks[-1] if blocks else None


def _extract_livecodebench_block(solution: str) -> str:
    lines = solution.split("\n")
    fences = [index for index, line in enumerate(lines) if "```" in line]
    if len(fences) < 2:
        return ""
    return "\n".join(lines[fences[-2] + 1 : fences[-1]])


def _partial_reward(results, *, binary: bool, power4: bool):
    if binary:
        return all(results)
    ratio = sum(results) / len(results)
    return ratio**4 if power4 else ratio


def _run_script_tests(solution: str, test_cases: Dict[str, Any], timeout: int, assert_key: str):
    results = []
    metadata = []
    for assertion in test_cases[assert_key]:
        program = IMPORT_PROMPT + solution + "\n" + assertion
        try:
            with _timeout(2):
                with tempfile.NamedTemporaryFile(mode="w", suffix=".py") as file:
                    file.write(program)
                    file.flush()
                    completed = subprocess.run(
                        ["python", file.name],
                        capture_output=True,
                        text=True,
                        timeout=timeout,
                    )
            passed = completed.returncode == 0
            results.append(passed)
            metadata.append("success" if passed else f"execution error: {completed.stderr}")
        except TimeoutError:
            results.append(False)
            metadata.append("code execution timed out")
        except Exception as error:
            results.append(False)
            metadata.append(f"execution exception: {error}")
    return results, metadata


def _score_check_based(
    solution: str,
    test_cases: Dict[str, Any],
    timeout: int,
    binary: bool,
    power4: bool,
):
    solution = test_cases["import_prefix"] + solution
    lines = [line for line in test_cases["test_code"].split("\n") if line]
    results = []
    metadata = []
    for index in range(1, len(lines)):
        program = solution + "\n" + lines[0] + lines[index]
        program += f"\ncheck({test_cases['entry_point']})"
        try:
            with _timeout(2):
                with tempfile.NamedTemporaryFile(mode="w", suffix=".py") as file:
                    file.write(program)
                    file.flush()
                    completed = subprocess.run(
                        ["python", file.name],
                        capture_output=True,
                        text=True,
                        timeout=timeout,
                    )
            passed = completed.returncode == 0
            results.append(passed)
            metadata.append("success" if passed else f"execution error: {completed.stderr}")
        except TimeoutError:
            results.append(False)
            metadata.append("code execution timed out")
        except Exception as error:
            results.append(False)
            metadata.append(f"execution exception: {error}")
    return _partial_reward(results, binary=binary, power4=power4), metadata


def _normalize_results(results):
    normalized = []
    for value in results:
        if isinstance(value, np.ndarray):
            value = value.item(0)
        if isinstance(value, np.bool_):
            value = bool(value)
        normalized.append(value)
    return normalized


def _livecodebench_sample(cache_path: str):
    problem = load_livecodebench_problem(cache_path)
    tests = list(problem.public_test_cases) + list(problem.private_test_cases)
    metadata = problem.metadata if isinstance(problem.metadata, dict) else json.loads(problem.metadata)
    return {
        "input_output": json.dumps(
            {
                "inputs": [test.input for test in tests],
                "outputs": [test.output for test in tests],
                "fn_name": metadata.get("func_name"),
            }
        )
    }


def compute_score(
    completion: str,
    test_cases: Dict[str, Any],
    *,
    livecodebench_data_dir: Optional[str] = None,
    timeout: int = 6,
    is_binary_reward: bool = True,
    is_power4_reward: bool = False,
):
    """Compute code reward with GePPO's extraction and grading semantics."""
    # GePPO takes the segment immediately after the first closing think tag.
    solution_text = completion.split("</think>")[1] if "</think>" in completion else completion

    if "question_id" in test_cases:
        try:
            cache_path = test_cases.get("cache_path")
            if not cache_path:
                if not livecodebench_data_dir:
                    raise ValueError("livecodebench_data_dir is required")
                cache_path = str(Path(livecodebench_data_dir) / f"{test_cases['question_id']}.pkl")
            extracted = _extract_livecodebench_block(solution_text)
            results, _ = check_correctness(
                _livecodebench_sample(cache_path),
                extracted,
                timeout=timeout,
                debug=False,
            )
            return all(value > 0 for value in results), None
        except Exception:
            traceback.print_exc(limit=10)
            return False, None

    if "import_prefix" in test_cases:
        solution = _extract_python_block(solution_text)
        if solution is None:
            return False, None
        try:
            ast.parse(solution)
            return _score_check_based(
                solution,
                test_cases,
                timeout,
                is_binary_reward,
                is_power4_reward,
            )
        except Exception as error:
            traceback.print_exc(limit=10)
            return False, f"code parsing error: {error}"

    if "inputs" in test_cases:
        solution = _extract_python_block(solution_text)
        if solution is None:
            return False, None
        try:
            ast.parse(solution)
            input_output = test_cases
            serialized = json.dumps(test_cases)
            if "fn_name" in input_output and "class Solution" not in solution:
                solution = _convert_function_to_class_method(solution, input_output["fn_name"])
                if solution is None:
                    return False, None
            results, metadata = check_correctness(
                {"input_output": serialized},
                solution,
                debug=False,
                timeout=timeout,
            )
            results = _normalize_results(results)
            if is_binary_reward:
                return sum(results) == len(results), [results, metadata]
            passed = sum(value if value in (False, True) else False for value in results)
            ratio = passed / len(results)
            return (ratio**4 if is_power4_reward else ratio), [results, metadata]
        except Exception:
            traceback.print_exc(limit=10)
            return False, None

    if "assert_case" in test_cases:
        solution = _extract_python_block(solution_text)
        if solution is None:
            return False, None
        try:
            ast.parse(solution)
            results, metadata = _run_script_tests(solution, test_cases, timeout, "assert_case")
            return _partial_reward(
                results,
                binary=is_binary_reward,
                power4=is_power4_reward,
            ), metadata
        except Exception as error:
            traceback.print_exc(limit=10)
            return False, f"code parsing error: {error}"

    raise ValueError(f"Unsupported GePPO code test schema: {sorted(test_cases)}")
