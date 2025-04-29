from abc import abstractmethod
import numpy as np

class scheduler():
    """
     所有学习率调度器的基类，定义基本接口
     """
    def __init__(self, optimizer) -> None:
        self.optimizer = optimizer
        self.step_count = 0
    
    @abstractmethod
    def step(self):
        pass


class StepLR(scheduler):
    """
    每间隔 step_size 个 epoch，将学习率乘以 gamma
    例如：每 30 epoch 乘 0.1
    """

    def __init__(self, optimizer, step_size=30, gamma=0.1) -> None:
        super().__init__(optimizer)
        self.step_size = step_size
        self.gamma = gamma

    def step(self) -> None:
        self.step_count += 1
        if self.step_count >= self.step_size:
            self.optimizer.init_lr *= self.gamma
            self.step_count = 0
# mynn/lr_scheduler.py
class MultiStepLR:
    def __init__(self, optimizer, milestones, gamma=0.5):
        self.optimizer = optimizer
        self.milestones = milestones  # 切换学习率的迭代数
        self.gamma = gamma
        self.step_count = 0

    def step(self):
        self.step_count += 1
        if self.step_count in self.milestones:
            # ✅ 这里改成调整 self.optimizer.lr，不是 init_lr
            self.optimizer.lr *= self.gamma
            print(f"Learning rate decayed to {self.optimizer.lr:.6f}")


class ExponentialLR(scheduler):
    """
    指数衰减：每个 epoch 学习率乘以 gamma（指数缩小）
    例如：gamma=0.95，每轮变成原来的 95%
    """

    def __init__(self, optimizer, gamma=0.95) -> None:
        super().__init__(optimizer)
        self.gamma = gamma

    def step(self) -> None:
        self.step_count += 1
        self.optimizer.init_lr *= self.gamma
