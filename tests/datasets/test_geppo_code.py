import json
import pickle

from datasets import Dataset

from roll.configs.data_args import DataArguments
from roll.datasets.dataset import get_dataset
from roll.datasets.geppo_code import (
    _CodeGenerationProblem,
    _Test,
    _TestType,
    load_livecodebench_tests,
)


def test_geppo_training_pickle_is_normalized(tmp_path):
    train_path = tmp_path / "train.pkl"
    rows = [
        {
            "data_source": "train-code-taco-medium_hard",
            "prompt": [{"role": "user", "content": "Read two integers and print their sum."}],
            "ability": "code",
            "reward_model": {
                "ground_truth": {
                    "inputs": ["1 2\n", "20 22\n"],
                    "outputs": ["3\n", "42\n"],
                },
                "style": "rule",
            },
            "extra_info": {"index": 7},
        }
    ]
    with train_path.open("wb") as file:
        pickle.dump(rows, file)

    dataset = get_dataset(
        DataArguments(
            file_name=str(train_path),
            dataset_type="geppo_code",
            messages="messages",
        )
    )

    assert len(dataset) == 1
    assert dataset[0]["id"] == "7"
    assert dataset[0]["messages"] == rows[0]["prompt"]
    assert dataset[0]["test_cases"] == [
        {"stdin": "1 2\n", "expected_stdout": "3\n"},
        {"stdin": "20 22\n", "expected_stdout": "42\n"},
    ]
    assert dataset[0]["test_case_function"] == ""
    assert dataset[0]["tag"] == "geppo_code"
    assert json.loads(dataset[0]["geppo_ground_truth"]) == rows[0]["reward_model"]["ground_truth"]


def test_livecodebench_parquet_loads_question_cache(tmp_path):
    parquet_path = tmp_path / "livecodebench_sample.parquet"
    cache_dir = tmp_path / "question_cache"
    cache_dir.mkdir()

    Dataset.from_list(
        [
            {
                "data_source": "test-code-livecodebench-medium",
                "prompt": [{"role": "user", "content": "Return the smallest valid number."}],
                "ability": "code",
                "reward_model": {
                    "ground_truth": {"question_id": "q1"},
                    "question_id": "q1",
                    "style": "rule",
                },
                "extra_info": {"question_id": "q1", "difficulty": "medium"},
            }
        ]
    ).to_parquet(str(parquet_path))

    problem = _CodeGenerationProblem()
    problem.question_id = "q1"
    problem.public_test_cases = [_Test(input="5", output="7", testtype=_TestType.FUNCTIONAL)]
    problem.private_test_cases = [_Test(input="10", output="11", testtype=_TestType.FUNCTIONAL)]
    problem.metadata = {"func_name": "smallestNumber"}
    with (cache_dir / "q1.pkl").open("wb") as file:
        pickle.dump(problem, file)

    dataset = get_dataset(
        DataArguments(
            file_name=str(parquet_path),
            dataset_type="geppo_code",
            livecodebench_data_dir=str(cache_dir),
            messages="messages",
        )
    )

    assert len(dataset) == 1
    assert dataset[0]["id"] == "q1"
    assert dataset[0]["difficulty"] == "medium"
    assert dataset[0]["case_type"] == "livecodebench"
    assert dataset[0]["test_case_function"] == ""
    assert dataset[0]["test_cases"] == [
        {"question_id": "q1", "cache_path": str(cache_dir / "q1.pkl")},
    ]
    test_cases, function_name = load_livecodebench_tests(dataset[0]["test_cases"][0]["cache_path"])
    assert function_name == "smallestNumber"
    assert test_cases == [
        {"stdin": "5", "expected_stdout": "7"},
        {"stdin": "10", "expected_stdout": "11"},
    ]
    assert dataset[0]["tag"] == "geppo_code"
    assert dataset[0]["geppo_ground_truth"] == (
        '{"question_id": "q1", "cache_path": "' + str(cache_dir / "q1.pkl") + '"}'
    )


def test_geppo_check_based_tests_are_preserved(tmp_path):
    train_path = tmp_path / "leetcode.pkl"
    ground_truth = {
        "entry_point": "Solution().add",
        "import_prefix": "from typing import *\n",
        "test_code": "def check(candidate):\n    assert candidate(a=1, b=2) == 3\n",
    }
    with train_path.open("wb") as file:
        pickle.dump(
            [
                {
                    "data_source": "train-code-leetcode-Easy",
                    "prompt": [{"role": "user", "content": "Implement add."}],
                    "reward_model": {"ground_truth": ground_truth, "style": "rule"},
                    "extra_info": {"index": 11},
                }
            ],
            file,
        )

    dataset = get_dataset(
        DataArguments(file_name=str(train_path), dataset_type="geppo_code", messages="messages")
    )

    assert dataset[0]["case_type"] == "check_based"
    assert dataset[0]["test_cases"] == [
        {
            "assert_code": ground_truth["test_code"],
            "entry_point": ground_truth["entry_point"],
            "import_prefix": ground_truth["import_prefix"],
        }
    ]
