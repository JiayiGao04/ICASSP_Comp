import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from collections import Counter
from fourier_head import Fourier_Head
from spectral_pooling import SpectralPooling_layer

# Data Preprocessing: Sliding Window
def sliding_window(data, window_size, step):
    num_windows = (len(data) - window_size) // step + 1
    return [data[i:i + window_size] for i in range(0, num_windows * step, step)]

# Load and preprocess data
class SleepDataset(Dataset):
    def __init__(self, data, is_train=True, window_size=250, step=125):
        self.samples = []
        self.is_train = is_train

        # Apply sliding window and normalization
        for sample in data:
            accel = np.array(sample['accel'])
            accel = (accel - accel.mean(axis=0)) / (accel.std(axis=0) + 1e-8)  # Standardize
            windows = sliding_window(accel, window_size, step)
            
            if is_train:
                for window in windows:
                    self.samples.append({'accel': window, 'label': sample['label']})
            else:
                chunk_id = sample['chunk_id']
                for window in windows:
                    self.samples.append({'accel': window, 'chunk_id': chunk_id})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        accel = torch.tensor(sample['accel'], dtype=torch.float32).permute(1, 0)
        if self.is_train:
            label = torch.tensor(sample['label'], dtype=torch.long)
            return accel, label
        else:
            chunk_id = sample['chunk_id']
            return accel, chunk_id

def load_data(train_path, test_path, window_size=250, step=125):
    with open(train_path, 'r') as f:
        train_data = json.load(f)
    with open(test_path, 'r') as f:
        test_data = json.load(f)

    train_dataset = SleepDataset(train_data, is_train=True, window_size=window_size, step=step)
    test_dataset = SleepDataset(test_data, is_train=False, window_size=window_size, step=step)
    return train_dataset, test_dataset

# Model definition (unchanged)
class HybridClassifier(nn.Module):
    def __init__(self, input_channels=3, num_classes=2, conv_kernel_size=7, time_size=64, lstm_hidden_size=64, mlp_hidden_size=128, num_frequencies=10,  device="cuda"):
        super(HybridClassifier, self).__init__()
        self.conv = nn.Conv1d(input_channels, 32, kernel_size=conv_kernel_size, padding=(conv_kernel_size - 1) // 2)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(input_size=32, hidden_size=lstm_hidden_size, num_layers=1, batch_first=True, bidirectional=True)
        self.pool = SpectralPooling_layer(time_size)
        self.mlp = nn.Sequential(
            nn.Linear(lstm_hidden_size * 2, mlp_hidden_size),
            nn.ReLU(),
            nn.Linear(mlp_hidden_size, num_classes)
        )
        self.mlp1 = nn.Sequential(
            nn.Linear(lstm_hidden_size * 2, mlp_hidden_size),
            nn.ReLU())
        self.classification_head = Fourier_Head(
            mlp_hidden_size, num_classes, num_frequencies, device=device
        )

    def forward(self, x):
        x = self.relu(self.conv(x)) # batch * C(32) * T(250)
        x = self.pool(x) # batch * C(32) * T(64)
        x = x.permute(0, 2, 1)  # batch * T(64) * C(32)
        lstm_out, _ = self.lstm(x)  # batch * T(64) * C(lstm_hidden_size * 2)
        x = lstm_out[:, -1, :]
        #x = self.mlp(x)
        x = self.mlp1(x)
        x = self.classification_head(x)
        return x

# Training function with optimizer and scheduler improvements
def train_model(model, train_loader, val_loader, epochs=1000, lr=1e-3, patience=50, device='cuda', save_path="best_model.pth"):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10, verbose=True)

    best_val_accuracy = 0.0
    best_model_state = None
    patience_counter = 0

    for epoch in range(epochs):
        start_time = time.time()  # Start timing the epoch

        model.train()
        train_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation phase
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == y).sum().item()
                total += y.size(0)
        val_accuracy = correct / total

        # Learning rate adjustment
        scheduler.step(val_accuracy)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_state = model.state_dict()
            patience_counter = 0
            torch.save(best_model_state, save_path)
            print(f"Model saved with validation accuracy: {best_val_accuracy:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss / len(train_loader):.4f}, "
              f"Validation Accuracy: {val_accuracy:.4f}, Best Accuracy: {best_val_accuracy:.4f}")

        # End timing the epoch
        epoch_time = time.time() - start_time

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    return model


def train_model_1(model, train_loader, val_loader, epochs=1000, lr=1e-3, patience=50, device='cuda', save_path="best_model.pth"):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10, verbose=True)

    best_val_accuracy = 0.0
    best_model_state = None
    patience_counter = 0

    for epoch in range(epochs):
        start_time = time.time()  # Start timing the epoch

        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            # Calculate accuracy for training
            _, predicted = torch.max(outputs, 1)
            correct_train += (predicted == y).sum().item()
            total_train += y.size(0)

        train_accuracy = correct_train / total_train

        # Validation phase
        model.eval()
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                _, predicted = torch.max(outputs, 1)
                correct_val += (predicted == y).sum().item()
                total_val += y.size(0)
        val_accuracy = correct_val / total_val
        val_loss = train_loss / len(train_loader)  # Assuming similar loss behavior for validation

        # Learning rate adjustment
        scheduler.step(val_accuracy)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_state = model.state_dict()
            patience_counter = 0
            torch.save(best_model_state, save_path)
            print(f"Model saved with validation accuracy: {best_val_accuracy:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

        # End timing the epoch
        epoch_time = time.time() - start_time

        print(f"Epoch {epoch+1}/{epochs}, "
              f"Train Loss: {train_loss / len(train_loader):.4f}, "
              f"Train Accuracy: {train_accuracy:.4f}, "
              f"Validation Loss: {val_loss:.4f}, "
              f"Validation Accuracy: {val_accuracy:.4f}, "
              f"Best Validation Accuracy: {best_val_accuracy:.4f}, "
              f"Epoch Time: {epoch_time:.2f}s")

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    return model


# Prediction function (unchanged)
def predict(model, test_loader, output_path='predictions.csv', device='cuda'):
    model.eval()
    results = []
    with torch.no_grad():
        for x, chunk_ids in test_loader:
            x = x.to(device)
            outputs = model(x)
            _, predicted = torch.max(outputs, 1)
            results.extend(zip(chunk_ids.tolist(), predicted.tolist()))

    df = pd.DataFrame(results, columns=['chunk_id', 'label'])
    df.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")

# Main script
if __name__ == "__main__":
    train_path = "./train_transformed.json"
    test_path = "./streamingtest.json"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    train_dataset, test_dataset = load_data(train_path, test_path, window_size=250, step=125)
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    BATCH_SIZE = 128

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = HybridClassifier(input_channels=3, num_classes=2, device=device)

    save_path = "best_model_with_spec_pool_fourier_head.pth"
    trained_model = train_model_1(model, train_loader, val_loader, epochs=1000, patience=50, save_path=save_path)

    model.load_state_dict(torch.load(save_path))
    predict(model, test_loader, output_path='predictions.csv')

# 读取 CSV 文件
input_file = "./predictions.csv"  # 输入文件路径
output_file = "./predictions_unique.csv"  # 输出文件路径
df = pd.read_csv(input_file)

# 对每个 chunk_id 分组，取多数投票的 label
unique_predictions = []
for chunk_id, group in df.groupby('chunk_id'):
    labels = group['label'].tolist()
    # 使用 Counter 找到出现次数最多的 label
    most_common_label = Counter(labels).most_common(1)[0][0]
    unique_predictions.append({'chunk_id': chunk_id, 'label': most_common_label})

# 保存结果为新的 CSV 文件
unique_df = pd.DataFrame(unique_predictions)
unique_df.to_csv(output_file, index=False)

print(f"去重后的文件已保存至: {output_file}")
