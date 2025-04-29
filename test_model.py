import mynn as nn
import numpy as np
from struct import unpack
import gzip
import matplotlib.pyplot as plt
import pickle

# === 1. 加载 CNN 模型 ===
model = nn.models.Model_CNN()
model.load_model(r'.\best_models\best_model.pickle')  # ✅ 注意路径是你保存CNN模型的地方！

# === 2. 加载测试集 ===
test_images_path = r'.\dataset\MNIST\t10k-images-idx3-ubyte.gz'
test_labels_path = r'.\dataset\MNIST\t10k-labels-idx1-ubyte.gz'

with gzip.open(test_images_path, 'rb') as f:
    magic, num, rows, cols = unpack('>4I', f.read(16))
    test_imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28, 28)

with gzip.open(test_labels_path, 'rb') as f:
    magic, num = unpack('>2I', f.read(8))
    test_labs = np.frombuffer(f.read(), dtype=np.uint8)

# === 3. 预处理 ===
test_imgs = test_imgs.astype(np.float32) / 255.0  # 归一化
test_imgs = test_imgs.reshape(-1, 1, 28, 28)       # 加一维通道 (batch, 1, 28, 28)

# === 4. 测试模型 ===
logits = model(test_imgs)
print("Test accuracy:", nn.metric.accuracy(logits, test_labs))
