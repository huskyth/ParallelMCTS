import multiprocessing
import os
import time
from multiprocessing import Pool

import platform

# 获取当前操作系统
current_os = platform.system()


class ConcurrentProcess:
    def __init__(self, tasks_number: int):
        self.__results = []
        self.task_number = tasks_number

    # 回调函数（主进程收集结果）
    def _collect_result(self, result):
        self.__results.append(result)

    @property
    def result(self):
        return self.__results

    def _clear_result(self):
        self.__results = []

    def _error_handler(self, err):
        raise err

    def process(self, worker_task, work_param_list):
        self._clear_result()
        # 确保在Windows系统上正确创建进程
        if current_os == "Windows":
            multiprocessing.freeze_support()

        # 创建进程池（默认使用所有可用CPU核心）
        with Pool() as pool:
            # 为每个数据项提交异步任务
            for num in range(self.task_number):
                pool.apply_async(
                    func=worker_task,  # 工作函数
                    args=work_param_list[num],  # 参数
                    callback=self._collect_result,  # 结果回调
                    error_callback=self._error_handler

                )

            # 关闭进程池（不再接受新任务）
            pool.close()
            # 等待所有进程完成
            pool.join()


class OBJE:
    def __init__(self):
        self.test = 2


def work(ci, n):
    import random
    time.sleep(2)
    print(os.getpid())
    ci.test += n
    return ci.test


if __name__ == '__main__':
    ci = OBJE()
    cp = ConcurrentProcess(5)
    cp.process(work, [(ci, 0), (ci, 1), (ci, 2), (ci, 3), (ci, 4)])

    print("after process , returns:", *cp.result)
    print(ci.test)
