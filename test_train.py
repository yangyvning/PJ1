# test_train.py
import mynn as nn
from draw_tools.plot import plot
import numpy as np
from struct import unpack
import gzip
import matplotlib.pyplot as plt
import pickle

np.random.seed(309)

# === 加载 MNIST 数据集 ===
train_images_path = r'.\dataset\MNIST\train-images-idx3-ubyte.gz'
train_labels_path = r'.\dataset\MNIST\train-labels-idx1-ubyte.gz'

with gzip.open(train_images_path, 'rb') as f:
    magic, num, rows, cols = unpack('>4I', f.read(16))
    train_imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28, 28)

with gzip.open(train_labels_path, 'rb') as f:
    magic, num = unpack('>2I', f.read(8))
    train_labs = np.frombuffer(f.read(), dtype=np.uint8)

# === 数据增强：平移增强 ===
def augment_image(img):
    shift = np.random.randint(-2, 3, size=2)
    shifted = np.roll(img, shift[0], axis=0)
    shifted = np.roll(shifted, shift[1], axis=1)
    if shift[0] > 0:
        shifted[:shift[0], :] = 0
    elif shift[0] < 0:
        shifted[shift[0]:, :] = 0
    if shift[1] > 0:
        shifted[:, :shift[1]] = 0
    elif shift[1] < 0:
        shifted[:, shift[1]:] = 0
    return shifted

augmented_imgs = [augment_image(img) for img in train_imgs]
train_imgs = np.array(augmented_imgs)

# === 划分训练集与验证集 ===
idx = np.random.permutation(np.arange(num))
with open('idx.pickle', 'wb') as f:
    pickle.dump(idx, f)

train_imgs = train_imgs[idx]
train_labs = train_labs[idx]

# ✅ 留 5000 作为验证集
valid_imgs = train_imgs[:5000]
valid_labs = train_labs[:5000]
train_imgs = train_imgs[5000:]
train_labs = train_labs[5000:]

train_imgs = train_imgs.astype(np.float32) / 255.0
valid_imgs = valid_imgs.astype(np.float32) / 255.0
train_imgs = train_imgs.reshape(-1, 1, 28, 28)
valid_imgs = valid_imgs.reshape(-1, 1, 28, 28)

# === He初始化函数 ===
def he_init(shape):
    fan_in = shape[1] if len(shape) == 2 else np.prod(shape[1:])
    return np.random.randn(*shape) * np.sqrt(2. / fan_in)

# === 搭建新模型 ===
cnn_model = nn.models.Model_CNN(initialize_method=he_init)

# === 优化器 + 损失函数 + 学习率调度器 ===
optimizer = nn.optimizer.Adam(init_lr=0.001, model=cnn_model)
scheduler = nn.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[5, 10, 15], gamma=0.5)
loss_fn = nn.op.MultiCrossEntropyLoss(model=cnn_model, max_classes=train_labs.max() + 1)

# === 创建 Runner 开始训练 ===
runner = nn.runner.RunnerM(cnn_model, optimizer, nn.metric.accuracy, loss_fn, scheduler=scheduler)
runner.train([train_imgs, train_labs], [valid_imgs, valid_labs],
             num_epochs=20, log_iters=20, save_dir='./best_models', batch_size=64, early_stop_rounds=100)

# === 绘制损失与准确率曲线 ===
_, axes = plt.subplots(1, 2)
_.set_tight_layout(True)
plot(runner, axes)
plt.show()

# === 可视化卷积核 ===
def visualize_conv_kernels(W):
    kernels = W  # [out_channels, in_channels, kernel_size, kernel_size]
    for i in range(kernels.shape[0]):
        kernel = kernels[i, 0]  # 可视化第0个输入通道
        plt.imshow(kernel, cmap='gray')
        plt.title(f'Conv Kernel {i}')
        plt.colorbar()
        plt.show()

cnn_model.load_model('./best_models/best_model.pickle')
visualize_conv_kernels(cnn_model.conv1.W)
