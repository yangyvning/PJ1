from abc import abstractmethod
import numpy as np


class Optimizer:
    """
    优化器的抽象基类，所有优化器都继承它。
    包含：
      - init_lr：初始学习率
      - model：模型，包含所有的参数和梯度
    """
    def __init__(self, init_lr, model) -> None:
        self.init_lr = init_lr
        self.model = model

    @abstractmethod
    def step(self):
        pass


class SGD(Optimizer):
    """
    标准随机梯度下降优化器
    """
    def __init__(self, init_lr, model):
        super().__init__(init_lr, model)
    
    def step(self):
        for layer in self.model.layers:
            if layer.optimizable == True:
                for key in layer.params.keys():
                    if layer.weight_decay:
                        layer.params[key] *= (1 - self.init_lr * layer.weight_decay_lambda)
                    layer.params[key] = layer.params[key] - self.init_lr * layer.grads[key]


class MomentGD(Optimizer):
    """
    动量法的 SGD 优化器（Momentum Gradient Descent）
    在更新时考虑了上一轮的方向，帮助收敛更快
    参数：
        mu：动量因子（常设为 0.9）
    """
    def __init__(self, init_lr, model, mu):
        super().__init__(init_lr, model)
        self.mu = mu  # 动量因子
        self.velocity = {}  # 用于保存每个参数的“历史更新方向”

        # 初始化 velocity（和所有参数形状一致）
        for layer in self.model.layers:
            if layer.optimizable:
                self.velocity[layer] = {}
                for key in layer.params.keys():
                    self.velocity[layer][key] = np.zeros_like(layer.params[key])


def step(self):
    for layer in self.model.layers:
        if layer.optimizable:
            for key in layer.params.keys():
                # L2 正则化（权重衰减）
                if layer.weight_decay:
                    layer.params[key] *= (1 - self.init_lr * layer.weight_decay_lambda)

                # 更新 velocity：v = mu * v - lr * grad
                self.velocity[layer][key] = (
                        self.mu * self.velocity[layer][key] - self.init_lr * layer.grads[key]
                )

                # 参数更新：param += velocity
                layer.params[key] += self.velocity[layer][key]



class Adam:
    """
    Adam 优化器
    """
    def __init__(self, init_lr=0.001, model=None, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = init_lr
        self.model = model
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.m = {}
        self.v = {}

    def step(self):
        self.t += 1
        for layer in self.model.layers:
            if not layer.optimizable:
                continue
            for name, param in layer.params.items():
                grad = layer.grads[name]
                key = (id(layer), name)

                if key not in self.m:
                    self.m[key] = np.zeros_like(grad)
                    self.v[key] = np.zeros_like(grad)

                self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grad
                self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grad ** 2)

                m_hat = self.m[key] / (1 - self.beta1 ** self.t)
                v_hat = self.v[key] / (1 - self.beta2 ** self.t)

                update = -self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
                layer.params[name] += update
