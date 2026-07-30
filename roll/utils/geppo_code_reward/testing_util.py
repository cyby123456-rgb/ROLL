# Adapted from GePPO's vendored LiveCodeBench evaluator.
import ast
import builtins
import faulthandler
import json
import os
import platform
import signal
import sys
import time
from decimal import Decimal
from enum import Enum
from io import StringIO
from types import ModuleType
from unittest.mock import mock_open, patch


IMPORT_STRING = """from string import *
from re import *
from datetime import *
from collections import *
from heapq import *
from bisect import *
from copy import *
from math import *
from random import *
from statistics import *
from itertools import *
from functools import *
from operator import *
from io import *
from sys import *
from json import *
from builtins import *
from typing import *
import string
import re
import datetime
import collections
import heapq
import bisect
import copy
import math
import random
import statistics
import itertools
import functools
import operator
import io
import sys
import json
sys.setrecursionlimit(50000)
"""


class CodeType(Enum):
    CALL_BASED = 0
    STANDARD_INPUT = 1


class TimeoutException(Exception):
    pass


def _timeout_handler(signum, frame):
    del signum, frame
    raise TimeoutException


class _Capturing(list):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        self._stringio.close = lambda *args: None
        return self

    def __exit__(self, *args):
        self.append(self._stringio.getvalue())
        del self._stringio
        sys.stdout = self._stdout


def _truncate(value, length=300):
    if not isinstance(value, str):
        value = str(value)
    if len(value) <= length:
        return value
    return value[: length // 2] + "...(truncated) ..." + value[-length // 2 :]


def _clean_main_guard(code: str) -> str:
    try:
        tree = ast.parse(code)
        last_block = tree.body[-1]
        if isinstance(last_block, ast.If) and ast.unparse(last_block.test).strip() == "__name__ == '__main__'":
            return ast.unparse(tree.body[:-1]) + "\n" + ast.unparse(last_block.body)
    except Exception:
        pass
    return code


def _make_function(code: str) -> str:
    try:
        imports = []
        body = []
        tree = ast.parse(code)
        for statement in tree.body:
            (imports if isinstance(statement, (ast.Import, ast.ImportFrom)) else body).append(statement)
        function = ast.FunctionDef(
            name="wrapped_function",
            args=ast.arguments(
                posonlyargs=[],
                args=[],
                kwonlyargs=[],
                kw_defaults=[],
                defaults=[],
            ),
            body=body,
            decorator_list=[],
            lineno=-1,
        )
        return IMPORT_STRING + "\n" + ast.unparse(imports) + "\n" + ast.unparse(function)
    except Exception:
        return code


def _call_method(method, inputs):
    if isinstance(inputs, list):
        inputs = "\n".join(inputs)
    inputs_line_iterator = iter(inputs.split("\n"))

    @patch("builtins.open", mock_open(read_data=inputs))
    @patch("sys.stdin", StringIO(inputs))
    @patch("sys.stdin.readline", lambda *args: next(inputs_line_iterator))
    @patch("sys.stdin.readlines", lambda *args: inputs.split("\n"))
    @patch("sys.stdin.read", lambda *args: inputs)
    def inner(current_method):
        try:
            return current_method()
        except SystemExit:
            return None

    return inner(method)


def _compile_code(code: str, timeout: int):
    signal.alarm(timeout)
    try:
        module = ModuleType("tmp_sol", "")
        exec(code, module.__dict__)
        compiled = module.Solution() if "class Solution" in code else module
        assert compiled is not None
        return compiled
    finally:
        signal.alarm(0)


def _to_decimals(line: str):
    try:
        return True, [Decimal(element) for element in line.split()]
    except Exception:
        return False, []


def _stripped_lines(value: str):
    return [line.strip() for line in value.strip().split("\n")]


def _grade_call_based(code: str, inputs: list, outputs: list, function_name: str, timeout: int):
    compiled = _compile_code(IMPORT_STRING + "\n\n" + code, timeout)
    method = getattr(compiled, function_name, None)
    if method is None:
        return None

    parsed_inputs = [[json.loads(line) for line in value.split("\n")] for value in inputs]
    parsed_outputs = [json.loads(value) for value in outputs]
    total_execution = 0.0
    results = []
    for test_input, expected in zip(parsed_inputs, parsed_outputs):
        signal.alarm(timeout)
        faulthandler.enable()
        try:
            start = time.time()
            prediction = method(*test_input)
            total_execution += time.time() - start
            signal.alarm(0)
            if isinstance(prediction, tuple):
                prediction = list(prediction)
            passed = prediction == expected
            results.append(passed)
            if not passed:
                return results, {
                    "output": _truncate(prediction),
                    "inputs": _truncate(test_input),
                    "expected": _truncate(expected),
                    "error_code": -2,
                    "error_message": "Wrong Answer",
                }
        except Exception as error:
            signal.alarm(0)
            timed_out = "timeoutexception" in repr(error).lower()
            results.append(-3 if timed_out else -4)
            return results, {
                "error": repr(error),
                "error_code": -3 if timed_out else -4,
                "error_message": "Time Limit Exceeded" if timed_out else "Runtime Error",
                "inputs": _truncate(test_input),
                "expected": _truncate(expected),
            }
        finally:
            signal.alarm(0)
            faulthandler.disable()
    return results, {"execution time": total_execution}


def _grade_stdio(code: str, inputs: list, outputs: list, timeout: int):
    compiled = _compile_code(_make_function(_clean_main_guard(code)), timeout)
    method = getattr(compiled, "wrapped_function", None)
    if method is None:
        return None

    results = []
    total_execution = 0.0
    for test_input, expected in zip(inputs, outputs):
        signal.alarm(timeout)
        faulthandler.enable()
        with _Capturing() as captured:
            try:
                start = time.time()
                _call_method(method, test_input)
                total_execution += time.time() - start
                signal.alarm(0)
            except Exception as error:
                signal.alarm(0)
                timed_out = "timeoutexception" in repr(error).lower()
                results.append(-3 if timed_out else -4)
                return results, {
                    "error": repr(error),
                    "error_code": -3 if timed_out else -4,
                    "error_message": "Time Limit Exceeded" if timed_out else "Runtime Error",
                    "inputs": _truncate(test_input),
                    "expected": _truncate(expected),
                }
            finally:
                signal.alarm(0)
                faulthandler.disable()

        prediction = captured[0]
        prediction_lines = _stripped_lines(prediction)
        expected_lines = _stripped_lines(expected)
        wrong_answer = {
            "output": _truncate(prediction),
            "inputs": _truncate(test_input),
            "expected": _truncate(expected),
            "error_code": -2,
        }
        if len(prediction_lines) != len(expected_lines):
            results.append(-2)
            wrong_answer["error_message"] = "Wrong answer: mismatched output length"
            return results, wrong_answer

        for index, (prediction_line, expected_line) in enumerate(zip(prediction_lines, expected_lines)):
            if prediction_line == expected_line:
                continue
            wrong_answer["error_message"] = (
                f"Wrong answer at output_line_idx={index}: "
                f"{_truncate(prediction_line)} != {_truncate(expected_line)}"
            )
            prediction_ok, prediction_decimals = _to_decimals(prediction_line)
            expected_ok, expected_decimals = _to_decimals(expected_line)
            if not prediction_ok or not expected_ok or prediction_decimals != expected_decimals:
                results.append(-2)
                return results, wrong_answer
        results.append(True)
    return results, {"execution time": total_execution}


def run_test(sample, test=None, debug=False, timeout=6):
    signal.signal(signal.SIGALRM, _timeout_handler)
    _reliability_guard()
    input_output = json.loads(sample["input_output"])
    function_name = input_output.get("fn_name")
    code_type = CodeType.STANDARD_INPUT if function_name is None else CodeType.CALL_BASED
    if test is None:
        raise AssertionError("should not happen: test code is none")

    if code_type == CodeType.CALL_BASED:
        signal.alarm(timeout)
        try:
            result = _grade_call_based(
                test,
                input_output["inputs"],
                input_output["outputs"],
                function_name,
                timeout,
            )
            if result is None:
                return [-4], {"error_code": -4, "error_message": "Unable to extract function"}
            return result
        except Exception as error:
            return [-4], {"error_code": -4, "error_message": f"Error during testing: {error}"}
        finally:
            signal.alarm(0)

    signal.alarm(timeout)
    try:
        result = _grade_stdio(test, input_output["inputs"], input_output["outputs"], timeout)
        if result is None:
            return [-4], {"error_code": -4, "error_message": "Unable to extract function"}
        return result
    except Exception as error:
        return [-4], {"error_code": -4, "error_message": f"Error during testing: {error}"}
    finally:
        signal.alarm(0)


def _reliability_guard(maximum_memory_bytes=None):
    if maximum_memory_bytes is not None:
        import resource

        resource.setrlimit(resource.RLIMIT_AS, (maximum_memory_bytes, maximum_memory_bytes))
        resource.setrlimit(resource.RLIMIT_DATA, (maximum_memory_bytes, maximum_memory_bytes))
        if platform.uname().system != "Darwin":
            resource.setrlimit(resource.RLIMIT_STACK, (maximum_memory_bytes, maximum_memory_bytes))

    faulthandler.disable()
    builtins.quit = None
    os.environ["OMP_NUM_THREADS"] = "1"
    for name in (
        "kill", "system", "putenv", "remove", "removedirs", "rmdir", "fchdir",
        "setuid", "fork", "forkpty", "killpg", "rename", "renames", "truncate",
        "replace", "unlink", "fchmod", "fchown", "chmod", "chown", "chroot",
        "lchflags", "lchmod", "lchown", "getcwd", "chdir",
    ):
        if hasattr(os, name):
            setattr(os, name, None)

    import shutil
    import subprocess

    shutil.rmtree = None
    shutil.move = None
    shutil.chown = None
    subprocess.Popen = None
    builtins.help = None
    for module_name in ("ipdb", "joblib", "resource", "psutil", "tkinter"):
        sys.modules[module_name] = None
