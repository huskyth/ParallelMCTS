import os
from concurrent.futures import ProcessPoolExecutor, as_completed


class ConcurrentProcess:
    def __init__(self, tasks_number: int):
        self.task_number = tasks_number

    def process(self, *args):
        worker = args[0]
        result = []
        with ProcessPoolExecutor(max_workers=min(4, os.cpu_count())) as ppe:
            future_list = [ppe.submit(worker, i + args[1], *args[2:]) for i in range(self.task_number)]
            for item in as_completed(future_list):
                data_list = item.result()
                result.append(data_list)

        return result
