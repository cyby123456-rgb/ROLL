import importlib

import pytest


score_module = importlib.import_module("roll.utils.geppo_code_reward.compute_score")


def test_training_reward_requires_exact_python_fence(monkeypatch):
    called = False

    def fake_check(*args, **kwargs):
        nonlocal called
        called = True
        return [True], {}

    monkeypatch.setattr(score_module, "check_correctness", fake_check)
    score, metadata = score_module.compute_score(
        "print(input())",
        {"inputs": ["1"], "outputs": ["1"]},
    )

    assert score is False
    assert metadata is None
    assert called is False


def test_training_reward_uses_last_python_block(monkeypatch):
    def fake_check(sample, solution, **kwargs):
        assert solution == "print('right')\n"
        return [True], {"execution time": 0}

    monkeypatch.setattr(score_module, "check_correctness", fake_check)
    completion = """```python
print('wrong')
```
```python
print('right')
```"""

    score, _ = score_module.compute_score(
        completion,
        {"inputs": [""], "outputs": ["right\n"]},
    )

    assert score is True


def test_top_level_function_is_wrapped_in_solution(monkeypatch):
    def fake_check(sample, solution, **kwargs):
        assert "class Solution:" in solution
        assert "def add(self, a, b):" in solution
        return [True], {}

    monkeypatch.setattr(score_module, "check_correctness", fake_check)
    score, _ = score_module.compute_score(
        "```python\ndef add(a, b):\n    return a + b\n```",
        {
            "inputs": ["1\n2"],
            "outputs": ["3"],
            "fn_name": "add",
        },
    )

    assert score is True


@pytest.mark.parametrize(
    ("completion", "expected_code", "expected_score"),
    [
        ("print('raw')", "", False),
        (
            "text\n```python\nprint('first')\n```\nmore\n```\nprint('last')\n```",
            "print('last')",
            True,
        ),
    ],
)
def test_livecodebench_uses_final_fenced_block(
    monkeypatch,
    completion,
    expected_code,
    expected_score,
):
    monkeypatch.setattr(score_module, "_livecodebench_sample", lambda path: {"input_output": "{}"})

    def fake_check(sample, solution, **kwargs):
        assert solution == expected_code
        return [expected_score], {}

    monkeypatch.setattr(score_module, "check_correctness", fake_check)
    score, _ = score_module.compute_score(
        completion,
        {"question_id": "q1", "cache_path": "/unused/q1.pkl"},
    )

    assert score is expected_score


def test_partial_and_power4_rewards(monkeypatch):
    monkeypatch.setattr(
        score_module,
        "check_correctness",
        lambda *args, **kwargs: ([True, False, True, False], {}),
    )
    completion = "```python\nprint(1)\n```"
    tests = {"inputs": ["", "", "", ""], "outputs": ["1", "1", "1", "1"]}

    partial, _ = score_module.compute_score(completion, tests, is_binary_reward=False)
    power4, _ = score_module.compute_score(
        completion,
        tests,
        is_binary_reward=False,
        is_power4_reward=True,
    )

    assert partial == 0.5
    assert power4 == 0.5**4


def test_check_based_reward_executes_each_assertion():
    ground_truth = {
        "entry_point": "Solution().add",
        "import_prefix": "",
        "test_code": (
            "def check(candidate):\n"
            "    assert candidate(1, 2) == 3\n"
            "    assert candidate(-2, 5) == 3\n"
        ),
    }

    correct, _ = score_module.compute_score(
        "```python\nclass Solution:\n    def add(self, a, b):\n        return a + b\n```",
        ground_truth,
    )
    wrong, _ = score_module.compute_score(
        "```python\nclass Solution:\n    def add(self, a, b):\n        return a - b\n```",
        ground_truth,
    )

    assert correct is True
    assert wrong is False


def test_stdio_reward_runs_in_isolated_process():
    score, metadata = score_module.compute_score(
        "```python\na, b = map(int, input().split())\nprint(a + b)\n```",
        {
            "inputs": ["1 2\n", "20 22\n"],
            "outputs": ["3\n", "42\n"],
        },
    )

    assert score is True
    assert metadata[0] == [True, True]
