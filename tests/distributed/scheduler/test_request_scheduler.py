import asyncio
from concurrent.futures import Future

from roll.distributed.scheduler.generate_scheduler import RequestScheduler
from roll.distributed.scheduler.protocol import DataProto
from roll.utils.functionals import GenerateRequestType


class _CompletedRef:
    def __init__(self):
        self._future = Future()
        self._future.set_result(None)

    def future(self):
        return self._future


class _RemoteAddRequest:
    def __init__(self, worker):
        self.worker = worker

    def remote(self, command, data):
        assert command == GenerateRequestType.ADD
        self.worker.request_ids.append(data.meta_info["request_id"])
        response = DataProto(
            meta_info={
                "request_id": data.meta_info["request_id"],
                "marker": data.meta_info["marker"],
            }
        )
        asyncio.get_running_loop().create_task(self.worker.scheduler.report_response(response))
        return _CompletedRef()


class _Worker:
    def __init__(self):
        self.scheduler = None
        self.request_ids = []
        self.add_request = _RemoteAddRequest(self)


class _InferCluster:
    def __init__(self, world_size):
        self.world_size = world_size
        self.workers = [_Worker() for _ in range(world_size)]


def test_generate_raw_request_uses_scheduler_dispatch_and_returns_callback_payload():
    async def run_test():
        infer_cluster = _InferCluster(world_size=2)
        scheduler_cls = RequestScheduler.__ray_metadata__.modified_class
        scheduler = scheduler_cls(infer_cluster=infer_cluster, pipeline_config=None)
        for worker in infer_cluster.workers:
            worker.scheduler = scheduler

        requests = [
            DataProto(meta_info={"src_rank": index, "marker": index})
            for index in range(4)
        ]
        responses = await asyncio.gather(
            *(scheduler.generate_raw_request(request) for request in requests)
        )

        assert [response.meta_info["marker"] for response in responses] == [0, 1, 2, 3]
        assert [len(worker.request_ids) for worker in infer_cluster.workers] == [2, 2]
        assert scheduler.request_id_2_dp_rank == {}
        assert all(not requests for requests in scheduler.inflight_requests)

    asyncio.run(run_test())
