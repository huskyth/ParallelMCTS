from abc import ABC, abstractmethod

import torch


class AbstractState(ABC):

    @abstractmethod
    def get_current_player(self) -> int:
        """返回当前玩家"""
        return -1

    @abstractmethod
    def do_action(self, action: int) -> None:
        """执行动作"""
        pass

    @abstractmethod
    def is_end(self) -> (bool, int):
        return False, -1

    @abstractmethod
    def get_torch_state(self) -> torch.Tensor:
        pass

    @abstractmethod
    def get_legal_moves(self) -> list:
        pass
