# ✅ runner.py：添加 EarlyStopping 支持
import numpy as np
import os
from tqdm import tqdm

class RunnerM():
    def __init__(self, model, optimizer, metric, loss_fn, batch_size=32, scheduler=None):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.metric = metric
        self.scheduler = scheduler
        self.batch_size = batch_size

        self.train_scores = []
        self.dev_scores = []
        self.train_loss = []
        self.dev_loss = []

    def train(self, train_set, dev_set, **kwargs):
        num_epochs = kwargs.get("num_epochs", 0)
        log_iters = kwargs.get("log_iters", 100)
        save_dir = kwargs.get("save_dir", "best_model")
        early_stop_rounds = kwargs.get("early_stop_rounds", 3)

        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        best_score = 0
        no_improve_count = 0

        for epoch in range(num_epochs):
            X, y = train_set
            assert X.shape[0] == y.shape[0]

            idx = np.random.permutation(range(X.shape[0]))
            X = X[idx]
            y = y[idx]

            for iteration in tqdm(range(int(X.shape[0] / self.batch_size) + 1)):
                train_X = X[iteration * self.batch_size : (iteration+1) * self.batch_size]
                train_y = y[iteration * self.batch_size : (iteration+1) * self.batch_size]

                logits = self.model(train_X)
                trn_loss = self.loss_fn(logits, train_y)
                self.train_loss.append(trn_loss)
                trn_score = self.metric(logits, train_y)
                self.train_scores.append(trn_score)

                self.loss_fn.backward()
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()

                dev_score, dev_loss = self.evaluate(dev_set)
                self.dev_scores.append(dev_score)
                self.dev_loss.append(dev_loss)

                if (iteration) % log_iters == 0:
                    print(f"epoch: {epoch}, iteration: {iteration}")
                    print(f"[Train] loss: {trn_loss}, score: {trn_score}")
                    print(f"[Dev] loss: {dev_loss}, score: {dev_score}")

                if dev_score > best_score:
                    save_path = os.path.join(save_dir, 'best_model.pickle')
                    self.save_model(save_path)
                    print(f"best accuracy updated: {best_score:.5f} --> {dev_score:.5f}")
                    best_score = dev_score
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                    if no_improve_count >= early_stop_rounds:
                        print("Early stopping triggered.")
                        self.best_score = best_score
                        return

        self.best_score = best_score

    def evaluate(self, data_set):
        X, y = data_set
        batch_size = self.batch_size  # 复用训练时的 batch_size
        total_score = 0
        total_loss = 0
        total_samples = 0

        for i in range(0, X.shape[0], batch_size):
            X_batch = X[i:i + batch_size]
            y_batch = y[i:i + batch_size]

            logits = self.model(X_batch)
            loss = self.loss_fn(logits, y_batch)
            score = self.metric(logits, y_batch)

            total_loss += loss * X_batch.shape[0]
            total_score += score * X_batch.shape[0]
            total_samples += X_batch.shape[0]

        avg_loss = total_loss / total_samples
        avg_score = total_score / total_samples

        return avg_score, avg_loss

    def save_model(self, save_path):
        self.model.save_model(save_path)
