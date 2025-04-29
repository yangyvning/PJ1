from abc import abstractmethod
import numpy as np

class Layer():
    def __init__(self) -> None:
        self.optimizable = True

    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def backward(self):
        pass


class Dropout(Layer):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        self.mask = None
        self.training = True
        self.optimizable = False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        if self.training:
            self.mask = (np.random.rand(*X.shape) > self.p).astype(np.float32)
            return X * self.mask / (1.0 - self.p)
        else:
            return X

    def clear_grad(self):
        self.grads = {'W': None, 'b': None}

    def backward(self, grad):
        if self.training:
            return grad * self.mask / (1.0 - self.p)
        else:
            return grad

class Linear(Layer):
    def __init__(self, in_dim, out_dim, initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8):
        super().__init__()
        self.W = initialize_method((in_dim, out_dim))
        self.b = initialize_method((1, out_dim))
        self.grads = {'W': None, 'b': None}
        self.input = None
        self.params = {'W': self.W, 'b': self.b}
        self.weight_decay = weight_decay
        self.weight_decay_lambda = weight_decay_lambda

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input = X
        return np.dot(X, self.W) + self.b

    def backward(self, grad):
        self.grads['W'] = np.dot(self.input.T, grad)
        self.grads['b'] = np.sum(grad, axis=0, keepdims=True)
        if self.weight_decay:
            self.grads['W'] += self.weight_decay_lambda * self.W
        return np.dot(grad, self.W.T)

    def clear_grad(self):
        self.grads = {'W': None, 'b': None}

class conv2D(Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.W = initialize_method((out_channels, in_channels, kernel_size, kernel_size))
        self.b = initialize_method((out_channels,))
        self.grads = {'W': None, 'b': None}
        self.input = None
        self.params = {'W': self.W, 'b': self.b}

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input = X
        batch_size, in_c, H, W = X.shape
        k = self.kernel_size
        s = self.stride
        out_H = (H - k) // s + 1
        out_W = (W - k) // s + 1
        patches = []
        for i in range(0, H - k + 1, s):
            for j in range(0, W - k + 1, s):
                patch = X[:, :, i:i + k, j:j + k].reshape(batch_size, -1)
                patches.append(patch)
        self.X_col = np.stack(patches, axis=1)  # 缓存展开输入用于反向传播
        W_col = self.W.reshape(self.out_channels, -1)
        out = self.X_col @ W_col.T + self.b
        out = out.transpose(0, 2, 1).reshape(batch_size, self.out_channels, out_H, out_W)
        return out

    def backward(self, grads):
        X = self.input
        batch_size, in_c, H, W = X.shape
        k = self.kernel_size
        s = self.stride
        out_H = (H - k) // s + 1
        out_W = (W - k) // s + 1

        grads_reshaped = grads.transpose(0, 2, 3, 1).reshape(batch_size, -1, self.out_channels)
        dW = np.zeros_like(self.W)
        for b in range(batch_size):
            dW += (grads_reshaped[b].T @ self.X_col[b]).reshape(self.W.shape)
        db = np.sum(grads, axis=(0, 2, 3))

        W_col = self.W.reshape(self.out_channels, -1)
        dX_col = np.zeros_like(self.X_col)
        for b in range(batch_size):
            dX_col[b] = grads_reshaped[b] @ W_col

        dX = np.zeros_like(X)
        idx = 0
        for i in range(0, H - k + 1, s):
            for j in range(0, W - k + 1, s):
                patch = dX_col[:, idx, :].reshape(batch_size, in_c, k, k)
                dX[:, :, i:i + k, j:j + k] += patch
                idx += 1

        self.grads['W'] = dW
        self.grads['b'] = db
        return dX

    def clear_grad(self):
        self.grads = {'W': None, 'b': None}

class ReLU(Layer):
    def __init__(self) -> None:
        super().__init__()
        self.input = None
        self.optimizable = False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input = X
        return np.maximum(0, X)

    def backward(self, grads):
        return grads * (self.input > 0).astype(grads.dtype)
# === 新增 MaxPool2D 类===
class MaxPool2D(Layer):
    """
    最大池化层，支持 2x2 池化，stride=2
    """
    def __init__(self, kernel_size=2, stride=2):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.input = None
        self.optimizable = False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        """
        X: [batch_size, channels, H, W]
        """
        self.input = X
        batch_size, channels, H, W = X.shape
        k = self.kernel_size
        s = self.stride
        out_H = (H - k) // s + 1
        out_W = (W - k) // s + 1
        out = np.zeros((batch_size, channels, out_H, out_W))

        for i in range(out_H):
            for j in range(out_W):
                region = X[:, :, i * s:i * s + k, j * s:j * s + k]
                out[:, :, i, j] = np.max(region, axis=(2, 3))

        return out

    def backward(self, grad):
        batch_size, channels, H, W = self.input.shape
        k = self.kernel_size
        s = self.stride
        out_H = (H - k) // s + 1
        out_W = (W - k) // s + 1
        dX = np.zeros_like(self.input)

        for i in range(out_H):
            for j in range(out_W):
                region = self.input[:, :, i * s:i * s + k, j * s:j * s + k]
                max_region = np.max(region, axis=(2, 3), keepdims=True)
                mask = (region == max_region)
                dX[:, :, i * s:i * s + k, j * s:j * s + k] += mask * grad[:, :, i, j][:, :, None, None]

        return dX


# =====================
# 多类交叉熵损失 + Softmax
# =====================
class MultiCrossEntropyLoss(Layer):
    """
    A multi-cross-entropy loss layer, with Softmax layer in it, which could be cancelled by method cancel_softmax
    """
    def __init__(self, model = None, max_classes = 10) -> None:

        super().__init__()
        self.model = model  #初始化这个变量！
        self.max_classes = max_classes
        self.has_softmax = True
        self.preds = None
        self.labels = None
        self.grads = None
        self.optimizable = False

    def __call__(self, predicts, labels):
        return self.forward(predicts, labels)
    
    def forward(self, predicts, labels):
        """
        predicts: [batch_size, D]
        labels : [batch_size, ]
        This function generates the loss.
        """


        self.labels = labels
        exps = np.exp(predicts - np.max(predicts, axis=1, keepdims=True))
        self.preds = exps / np.sum(exps, axis=1, keepdims=True)

            # 交叉熵损失
        batch_size = predicts.shape[0]
        correct_class_probs = self.preds[np.arange(batch_size), labels]
        loss = -np.mean(np.log(correct_class_probs + 1e-10))  # 避免 log(0)

        return loss
    
    def backward(self):
        # first compute the grads from the loss to the input
        # / ---- your codes here ----/
        # Then send the grads to model for back propagation
        """
        输出 logits 的梯度，用于反向传播
        """
        batch_size = self.preds.shape[0]
        self.grads = self.preds.copy()
        self.grads[np.arange(batch_size), self.labels] -= 1
        self.grads /= batch_size
        self.model.backward(self.grads)

    def cancel_soft_max(self):
        self.has_softmax = False
        return self

    # =====================
    # L2正则化类（可选，已在 Linear 中支持）
    # =====================
class L2Regularization(Layer):
    """
    L2 Reg can act as weight decay that can be implemented in class Linear.
    """

    def __init__(self, lambda_=1e-4) -> None:
        super().__init__()
        self.lambda_ = lambda_
        self.optimizable = False

    def forward(self, weights):
        # 返回正则项损失
        return self.lambda_ * np.sum(weights ** 2)

    def backward(self, weights):
        # 返回正则项对 weights 的梯度
        return 2 * self.lambda_ * weights


# =====================
# softmax 函数（独立函数）
# =====================
def softmax(X):
    x_max = np.max(X, axis=1, keepdims=True)
    x_exp = np.exp(X - x_max)
    partition = np.sum(x_exp, axis=1, keepdims=True)
    return x_exp / partition


class Flatten(Layer):
    def __init__(self):
        super().__init__()
        self.input_shape = None
        self.optimizable = False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input_shape = X.shape
        return X.reshape(X.shape[0], -1)

    def backward(self, grad):
        return grad.reshape(self.input_shape)
