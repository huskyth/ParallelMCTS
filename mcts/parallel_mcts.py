import copy

import numpy as np
import asyncio
from chess.common import MOVE_TO_INDEX_DICT, MAX_STEPS
from mcts.node import Node
from asyncio import Queue
import torch

from network_wrapper import ChessNetWrapper


class Future:
    def __init__(self, future, state):
        self.future = future
        self.state = state


class MCTS:
    def __init__(self, predict, simulate_time=None):
        self.root = Node(1)
        self.model_predict = predict
        self.simulate_times = 1600 if simulate_time is None else simulate_time
        self.current_simulate = 0
        self.expanding_set = set()
        self.q = Queue(maxsize=400)
        self.loop = asyncio.get_event_loop()
        self.visual_loss_c = 3
        self.max_h = 0

    async def _simulate(self, state, idx):
        current_node = self.root
        counter = 0
        while True:
            while current_node in self.expanding_set:
                await asyncio.sleep(1e-3)
            if current_node.is_leaf() or counter > MAX_STEPS:
                break

            action, current_node = current_node.select(self.visual_loss_c)
            state.do_action(action)
            counter += 1
        if self.max_h < counter:
            self.max_h = counter

        is_end, winner = state.is_end()
        if is_end is True:
            assert winner is not None
            value = 1 if winner == state.get_current_player() else -1
            current_node.update(-value)
        else:
            if counter > MAX_STEPS:
                print(f"COUNT LARGE THAN {MAX_STEPS}")
                current_node.update(0)
            else:
                self.expanding_set.add(current_node)
                value, probability = await self.predict(state.get_torch_state())
                available_action = state.get_legal_moves(state.get_current_player())

                available_ = set()
                for move in available_action:
                    available_.add(MOVE_TO_INDEX_DICT[move])
                for idx, p in enumerate(probability):
                    if idx not in available_:
                        probability[idx] = 0
                probability /= probability.sum()
                # TODO: test it
                current_node.expand(probability)
                current_node.update(-value)
                self.expanding_set.remove(current_node)

        self.current_simulate += 1

    async def push_queue(self, state):
        future = self.loop.create_future()
        future = Future(future, state)
        await self.q.put(future)
        return future

    async def handle(self):
        while self.current_simulate != self.simulate_times:
            if self.q.qsize() <= 0:
                await asyncio.sleep(1e-3)
                continue
            item_list = [self.q.get_nowait() for _ in range(self.q.qsize())]
            state_list = [x.state for x in item_list]
            state = torch.cat(state_list, dim=0)
            v, p = self.model_predict(state)
            for v_item, p_item, future_item in zip(v, p, item_list):
                future_item.future.set_result((v_item.item(), p_item))

    async def predict(self, state):
        future = await self.push_queue(state)
        return await future.future

    def update_tree(self, move):
        if move not in self.root.children:
            self.root = Node(1)
        else:
            if self.root.children[move].p > 0:
                self.root = self.root.children[move]
                self.root.parent = None
            else:
                self.root = Node(1)

    def get_action_probability(self, state, is_greedy):
        coroutine_list = []
        self.current_simulate = 0
        self.max_h = 0
        for i in range(self.simulate_times):
            state_copy = copy.deepcopy(state)
            temp = self._simulate(state_copy, i)
            coroutine_list.append(temp)
        coroutine_list.append(self.handle())
        self.loop.run_until_complete(asyncio.gather(*coroutine_list))
        probability = np.array([item.visit for item in self.root.children.values()])

        if is_greedy:
            max_visit = np.max(probability)
            probability = np.where(probability == max_visit, 1, 0)

        visit_list = probability / probability.sum()
        return visit_list


if __name__ == '__main__':
    n = ChessNetWrapper()
    m = MCTS(n.predict)
    state_test = torch.randn((8, 1, 7, 7))
    y = m.model_predict(state_test)
    for v, p in zip(*m.model_predict(state_test)):
        print(v, p)
    print(y)
    a = np.arange(8).reshape(8, 1)
    b = np.arange(8 * 72).reshape(8, 72)
    for v, p in zip(a, b):
        print(v, p)
