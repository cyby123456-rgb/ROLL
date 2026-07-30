# Adapted from GePPO's vendored LiveCodeBench evaluator.
import json
import multiprocessing
import time

import psutil

from roll.utils.geppo_code_reward.testing_util import run_test


def _run_in_child(sample, generation, debug, connection, timeout):
    try:
        test_result, test_metadata = run_test(
            sample,
            test=generation,
            debug=debug,
            timeout=timeout,
        )
        connection.send((test_result, test_metadata))
    finally:
        connection.close()


def check_correctness(sample, generation, timeout, debug=True, max_memory_gb=50):
    """Run one generated program with GePPO's timeout and memory watchdog."""
    receive_connection, send_connection = multiprocessing.Pipe(duplex=False)
    process = multiprocessing.Process(
        target=_run_in_child,
        args=(sample, generation, debug, send_connection, timeout),
    )
    process.start()
    send_connection.close()

    memory_threshold = max_memory_gb * 1024 * 1024 * 1024
    inputs = json.loads(sample["input_output"])["inputs"]
    timeout_seconds = min((timeout + 1) * len(inputs) + 5, 300)
    start = time.time()
    while process.is_alive():
        try:
            monitored = psutil.Process(process.pid)
            memory_used = monitored.memory_info().rss
            for child in monitored.children(recursive=True):
                try:
                    memory_used += child.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            if memory_used > memory_threshold:
                process.kill()
                break
        except psutil.NoSuchProcess:
            break

        if time.time() - start > timeout_seconds:
            process.kill()
            break
        time.sleep(0.1)

    process.join(1)
    try:
        if receive_connection.poll():
            return receive_connection.recv()
    except EOFError:
        pass
    finally:
        receive_connection.close()
    return [-1 for _ in inputs], {}
