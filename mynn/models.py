# models.py
from .op import *
import pickle

class MaxPool2D(Layer):
    def __init__(self, kernel_size=2, stride=2):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.input = None
        self.optimizable = False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input = X
        batch_size, c, h, w = X.shape
        new_h = (h - self.kernel_size) // self.stride + 1
        new_w = (w - self.kernel_size) // self.stride + 1

        output = np.zeros((batch_size, c, new_h, new_w))
        for i in range(new_h):
            for j in range(new_w):
                h_start = i * self.stride
                w_start = j * self.stride
                region = X[:, :, h_start:h_start+self.kernel_size, w_start:w_start+self.kernel_size]
                output[:, :, i, j] = np.max(region, axis=(2,3))
        return output

    def backward(self, grad_output):
        X = self.input
        batch_size, c, h, w = X.shape
        new_h = (h - self.kernel_size) // self.stride + 1
        new_w = (w - self.kernel_size) // self.stride + 1

        grad_input = np.zeros_like(X)

        for i in range(new_h):
            for j in range(new_w):
                h_start = i * self.stride
                w_start = j * self.stride
                region = X[:, :, h_start:h_start+self.kernel_size, w_start:w_start+self.kernel_size]
                max_region = np.max(region, axis=(2,3), keepdims=True)
                mask = (region == max_region)
                grad_input[:, :, h_start:h_start+self.kernel_size, w_start:w_start+self.kernel_size] += mask * grad_output[:, :, i, j][:, :, None, None]

        return grad_input

class Model_CNN(Layer):
    def __init__(self, initialize_method=np.random.normal):
        super().__init__()
        self.conv1 = conv2D(1, 16, 3, initialize_method=initialize_method)  # 16×26×26
        self.act1 = ReLU()
        self.pool1 = MaxPool2D(2, 2)  # 16×13×13

        self.conv2 = conv2D(16, 32, 3, initialize_method=initialize_method)  # 32×11×11
        self.act2 = ReLU()
        self.pool2 = MaxPool2D(2, 2)  # 32×5×5

        self.flatten = Flatten()
        self.fc1 = Linear(32 * 5 * 5, 128, initialize_method=initialize_method)
        self.act3 = ReLU()
        self.drop = Dropout(p=0.3)
        self.fc2 = Linear(128, 10, initialize_method=initialize_method)

        self.layers = [
            self.conv1, self.act1, self.pool1,
            self.conv2, self.act2, self.pool2,
            self.flatten,
            self.fc1, self.act3, self.drop,
            self.fc2
        ]

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        out = X
        for layer in self.layers:
            out = layer(out)
        return out

    def backward(self, loss_grad):
        grads = loss_grad
        for layer in reversed(self.layers):
            grads = layer.backward(grads)
        return grads

    def save_model(self, save_path):
        param_list = [None, None]
        for layer in self.layers:
            if layer.optimizable:
                param_list.append({
                    'W': layer.params['W'],
                    'b': layer.params['b'],
                    'weight_decay': getattr(layer, 'weight_decay', False),
                    'lambda': getattr(layer, 'weight_decay_lambda', 0.0)
                })
        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)

    def load_model(self, save_path):
        with open(save_path, 'rb') as f:
            param_list = pickle.load(f)
        idx = 2
        for layer in self.layers:
            if layer.optimizable:
                layer.W = param_list[idx]['W']
                layer.b = param_list[idx]['b']
                layer.params['W'] = layer.W
                layer.params['b'] = layer.b
                layer.weight_decay = param_list[idx]['weight_decay']
                layer.weight_decay_lambda = param_list[idx]['lambda']
                idx += 1
